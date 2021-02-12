import tensorflow as tf

from bnn.regression_algo import RegressionModel
from modules.neural_network import BatchedFullyConnectedNN
from modules.prior_posterior import GaussianPosterior, GaussianPrior
from modules.likelihood import GaussianLikelihood


class BayesianNeuralNetworkVI(RegressionModel):

    def __init__(self, x_train, y_train, hidden_layer_sizes=(32, 32), activation='elu',
                 likelihood_std=0.1, learn_likelihood=True, prior_std=1.0, prior_weight=0.1,
                 likelihood_prior_mean=tf.math.log(0.1), likelihood_prior_std=1.0,
                 batch_size_vi=10, batch_size=8, lr=1e-3):

        self.prior_weight = prior_weight
        self.likelihood_std = likelihood_std
        self.batch_size = batch_size
        self.batch_size_vi = batch_size_vi

        # data handling
        self._process_train_data(x_train, y_train)

        # setup nn
        self.nn = BatchedFullyConnectedNN(self.batch_size_vi, self.output_dim, hidden_layer_sizes, activation)
        self.nn.build((None, self.input_dim))

        # setup prior
        self.nn_param_size = self.nn.get_variables_stacked_per_model().shape[-1]
        if learn_likelihood:
            self.likelihood_param_size = self.output_dim
        else:
            self.likelihood_param_size = 0
        self.prior = GaussianPrior(self.nn_param_size, nn_prior_std=prior_std,
                                   likelihood_param_size=self.likelihood_param_size,
                                   likelihood_prior_mean=likelihood_prior_mean,
                                   likelihood_prior_std=likelihood_prior_std)

        # Likelihood
        self.likelihood = GaussianLikelihood(self.output_dim, self.batch_size_vi)

        # setup posterior
        self.posterior = GaussianPosterior(self.nn.get_variables_stacked_per_model(), self.likelihood_param_size)

        # setup optimizer
        self.optim = tf.keras.optimizers.Adam(lr)

    def predict(self, x, num_posterior_samples=20):
        # data handling
        x = self._handle_input_data(x, convert_to_tensor=True)
        x = self._normalize_data(x)

        # nn prediction
        y_pred_batches = []
        likelihood_std_batches = []
        for _ in range(num_posterior_samples // self.batch_size_vi):
            sampled_params = self.posterior.sample((self.batch_size_vi,))
            nn_params, likelihood_std = self._split_into_nn_params_and_likelihood_std(sampled_params)
            likelihood_std_batches.append(likelihood_std)
            y_pred_batches.append(self.nn.call_parametrized(x, nn_params))
        y_pred = tf.concat(y_pred_batches, axis=0)
        likelihood_std = tf.concat(likelihood_std_batches, axis=0)

        pred_dist = self.likelihood.get_pred_mixture_dist(y_pred, likelihood_std)

        # unnormalize preds
        y_pred = self._unnormalize_preds(y_pred)
        pred_dist = self._unnormalize_predictive_dist(pred_dist)
        return y_pred, pred_dist

    @tf.function
    def step(self, x_batch: tf.Tensor, y_batch: tf.Tensor) -> tf.Tensor:
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            # keep in mind: len(trainable variables) = number of defined Variables in class and all parent classes
            tape.watch(self.posterior.trainable_variables)

            # sample batch of parameters from the posterior
            sampled_params = self.posterior.sample((self.batch_size_vi,))
            nn_params, likelihood_std = self._split_into_nn_params_and_likelihood_std(sampled_params)

            # compute log-likelihood
            y_pred = self.nn.call_parametrized(x_batch, nn_params)  # (batch_size_vi, batch_size, 1)
            avg_log_likelihood = self.likelihood.log_prob(y_pred, y_batch, likelihood_std)

            # compute kl
            kl_divergence = self.posterior.log_prob(sampled_params) - self.prior.log_prob(sampled_params)
            avg_kl_divergence = tf.reduce_mean(kl_divergence) / self.num_train_samples

            # compute elbo
            elbo = - avg_log_likelihood + avg_kl_divergence * self.prior_weight

        # compute gradient of elbo wrt posterior parameters
        grads = tape.gradient(elbo, self.posterior.trainable_variables)
        self.optim.apply_gradients(zip(grads, self.posterior.trainable_variables))
        return elbo


if __name__ == '__main__':
    import numpy as np

    np.random.seed(0)
    tf.random.set_seed(0)

    d = 1  # dimensionality of the data

    n_train = 50
    x_train = np.random.uniform(-4, 4, size=(n_train, d))
    y_train = np.sin(x_train) + np.random.normal(scale=0.1, size=x_train.shape)

    n_val = 200
    x_val = np.random.uniform(-4, 4, size=(n_val, d))
    y_val = np.sin(x_val) + np.random.normal(scale=0.1, size=x_val.shape)

    nn = BayesianNeuralNetworkVI(x_train, y_train, hidden_layer_sizes=(32, 32), prior_weight=0.001, learn_likelihood=False)

    n_iter_fit = 2000
    for i in range(10):
        nn.fit(x_val=x_val, y_val=y_val, log_period=10, num_iter_fit=n_iter_fit)
        if d == 1:
            x_plot = tf.range(-8, 8, 0.1)
            nn.plot_predictions(x_plot, iteration=(i + 1) * n_iter_fit, experiment="bnn_svgd", show=True)
