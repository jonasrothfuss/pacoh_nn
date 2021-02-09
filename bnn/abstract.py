import tensorflow as tf
import tensorflow_probability as tfp
from tqdm import trange

from modules.affine_transform import AffineTransform
from modules.data_sampler import MetaDatasetSampler

tfd = tfp.distributions

class MetaRegressionModel:

    def _process_meta_train_data(self, meta_train_data, meta_batch_size, batch_size, n_batched_models_train):
        self.num_meta_train_tasks = len(meta_train_data)
        self.meta_train_sampler = MetaDatasetSampler(meta_train_data, batch_size, meta_batch_size=meta_batch_size,
                                                    n_batched_models=n_batched_models_train, tiled=True)
        self.x_mean, self.y_mean, self.x_std, self.y_std = self.meta_train_sampler.get_standardization_stats()
        self.input_dim = self.meta_train_sampler.input_dim
        self.output_dim = self.meta_train_sampler.output_dim

class RegressionModel:
    likelihood = None
    batch_size = None
    nn_param_size = None
    likelihood_param_size = None
    likelihood_std = None

    def fit(self, x_val=None, y_val=None, log_period=500, num_iter_fit=None):
        train_batch_sampler = self._get_batch_sampler(self.x_train, self.y_train, self.batch_size)
        loss_list = []
        pbar = trange(num_iter_fit)
        for i in pbar:
            x_batch, y_batch = next(train_batch_sampler)
            loss = self.step(x_batch, y_batch)
            loss_list.append(loss)

            if i % log_period == 0:
                loss = tf.reduce_mean(tf.convert_to_tensor(loss_list)).numpy()
                loss_list = []
                message = dict(loss=loss)
                if x_val is not None and y_val is not None:
                    metric_dict = self.eval(x_val, y_val)
                    message.update(metric_dict)
                pbar.set_postfix(message)

    def eval(self, x, y):
        x, y = self._handle_input_data(x, y, convert_to_tensor=True)
        _, pred_dist = self.predict(x)
        return self.likelihood.calculate_eval_metrics(pred_dist, y)

    def plot_predictions(self, x_plot, iteration=None, experiment=None, show=False):
        from matplotlib import pyplot as plt
        assert self.input_size == 1 and self.output_size == 1
        y_pred, pred_dist = self.predict(x_plot)
        fig, ax = plt.subplots(1, 1)

        # plot predictive mean and confidence interval
        ax.plot(x_plot, pred_dist.mean())
        lcb, ucb = pred_dist.mean() - 2 * pred_dist.stddev(), pred_dist.mean() + 2 * pred_dist.stddev()
        ax.fill_between(x_plot, lcb.numpy().flatten(), ucb.numpy().flatten(), alpha=0.2)

        for i in range(y_pred.shape[0]):
            plt.plot(x_plot, y_pred[i], color='green', alpha=0.4, linewidth=1.0)
        plt.show()

        # unnormalize training data & plot it
        x_train = self.x_train * self.x_std + self.x_mean
        y_train = self.y_train * self.y_std + self.y_mean
        ax.scatter(x_train, y_train, label="training")
        if show:
            fig.show()
        if experiment:
            import os
            parent_path = "../data/figures/" + experiment
            os.makedirs(parent_path, exist_ok=True)
            plt.savefig(parent_path + f"/{iteration}.pdf")
        plt.close()

    def _process_train_data(self, x_train, y_train):
        self.x_train, self.y_train = self._handle_input_data(x_train, y_train, convert_to_tensor=True)
        self.input_size, self.output_size = self.x_train.shape[-1], self.y_train.shape[-1]
        self.num_train_samples = self.x_train.shape[0]
        self._compute_normalization_stats(self.x_train, self.y_train)
        self.x_train, self.y_train = self._normalize_data(self.x_train, self.y_train)

    def _compute_normalization_stats(self, x_train, y_train):
        self.x_mean = tf.reduce_mean(x_train, axis=0)
        self.x_std = tfp.stats.stddev(x_train, sample_axis=0)
        self.y_mean = tf.reduce_mean(y_train, axis=0)
        self.y_std = tfp.stats.stddev(y_train, sample_axis=0)
        self.affine_pred_dist_transform = AffineTransform(normalization_mean=self.y_mean,
                                                          normalization_std=self.y_std)

    def _get_batch_sampler(self, x, y, batch_size):
        x, y = self._handle_input_data(x, y, convert_to_tensor=True)
        num_train_points = x.shape[0]

        if batch_size == -1:
            batch_size = num_train_points
        elif batch_size > 0:
            pass
        else:
            raise AssertionError('batch size must be either positive or -1')

        train_dataset = tf.data.Dataset.from_tensor_slices((x, y))
        train_dataset = train_dataset.shuffle(buffer_size=num_train_points, reshuffle_each_iteration=True)
        train_dataset = train_dataset.repeat()
        train_dataset = train_dataset.batch(batch_size)
        train_batch_sampler = train_dataset.__iter__()
        return train_batch_sampler

    def _normalize_data(self, x, y=None):
        x = (x - self.x_mean) / self.x_std
        if y is None:
            return x
        else:
            y = (y - self.y_mean) / self.y_std
            return x, y

    def _unnormalize_preds(self, y):
        return y * self.y_std + self.y_mean

    def _unnormalize_predictive_dist(self, pred_dist):
        return self.affine_pred_dist_transform.apply(pred_dist)

    @staticmethod
    def _handle_input_data(x, y=None, convert_to_tensor=True, dtype=tf.float32):
        if x.ndim == 1:
            x = tf.expand_dims(x, -1)

        assert x.ndim == 2

        if y is not None:
            if y.ndim == 1:
                y = tf.expand_dims(y, -1)
            assert x.shape[0] == y.shape[0]
            assert y.ndim == 2

            if convert_to_tensor:
                x, y = tf.cast(x, dtype=dtype), tf.cast(y, dtype=dtype)
            return x, y
        else:
            if convert_to_tensor:
                x = tf.cast(x, dtype=dtype)
            return x

    def _split_into_nn_params_and_likelihood_std(self, params):
        tf.assert_equal(tf.rank(params), 2)
        tf.assert_equal(params.shape[-1], self.nn_param_size + self.likelihood_param_size)
        n_particles = params.shape[0]
        nn_params = params[:, :self.nn_param_size]
        if self.likelihood_param_size > 0:
            likelihood_std = tf.exp(params[:, -self.likelihood_param_size:])
        else:
            likelihood_std = tf.ones((n_particles, self.output_size)) * self.likelihood_std

        tf.assert_equal(likelihood_std.shape, (n_particles, self.output_size))
        return nn_params, likelihood_std

    def predict(self, x):
        raise NotImplementedError

    def step(self, x_batch, y_batch):
        raise NotImplementedError
