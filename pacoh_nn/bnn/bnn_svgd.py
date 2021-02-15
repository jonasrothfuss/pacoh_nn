import tensorflow as tf
import tensorflow_probability as tfp
import math

from pacoh_nn.bnn.regression_algo import RegressionModel
from pacoh_nn.modules.neural_network import BatchedFullyConnectedNN
from pacoh_nn.modules.prior_posterior import GaussianPrior
from pacoh_nn.modules.likelihood import GaussianLikelihood

tfd = tfp.distributions
tfk = tfp.math.psd_kernels


class BayesianNeuralNetworkSVGD(RegressionModel):

    def __init__(self, x_train, y_train, hidden_layer_sizes=(32, 32, 32, 32), activation='relu',
                 likelihood_std=0.1, learn_likelihood=True, prior_std=0.1, prior_weight=1e-4,
                 likelihood_prior_mean=tf.math.log(0.1), likelihood_prior_std=1.0, sqrt_mode=False,
                 n_particles=10, batch_size=8, bandwidth=100., lr=1e-3, meta_learned_prior=None,
                 normalization_stats=None):

        self.prior_weight = prior_weight
        self.likelihood_std = likelihood_std
        self.batch_size = batch_size
        self.n_particles = n_particles
        self.sqrt_mode = sqrt_mode

        # data handling
        self._process_train_data(x_train, y_train, normalization_stats)

        # setup nn
        self.nn = BatchedFullyConnectedNN(n_particles, self.output_dim, hidden_layer_sizes, activation)
        self.nn.build((None, self.input_dim))

        # setup prior
        self.nn_param_size = self.nn.get_variables_stacked_per_model().shape[-1]
        if learn_likelihood:
            self.likelihood_param_size = self.output_dim
        else:
            self.likelihood_param_size = 0

        if meta_learned_prior is None:
            self.prior = GaussianPrior(self.nn_param_size, nn_prior_std=prior_std,
                                       likelihood_param_size=self.likelihood_param_size,
                                       likelihood_prior_mean=likelihood_prior_mean,
                                       likelihood_prior_std=likelihood_prior_std)
            self.meta_learned_prior_mode = False
        else:
            self.prior = meta_learned_prior
            assert meta_learned_prior.get_variables_stacked_per_model().shape[-1] == \
                   2 * (self.nn_param_size + self.likelihood_param_size)
            assert n_particles % self.prior.n_batched_priors == 0, "n_particles must be multiple of n_batched_priors"
            self.meta_learned_prior_mode = True

        # Likelihood
        self.likelihood = GaussianLikelihood(self.output_dim, n_particles)

        # setup particles
        if self.meta_learned_prior_mode:
            # initialize posterior particles from meta-learned prior
            params = tf.reshape(self.prior.sample(n_particles // self.prior.n_batched_priors), (n_particles, -1))
            self.particles = tf.Variable(params)
        else:
            # initialize posterior particles from model initialization
            nn_params = self.nn.get_variables_stacked_per_model()
            likelihood_params = tf.ones((self.n_particles, self.likelihood_param_size)) * likelihood_prior_mean
            self.particles = tf.Variable(tf.concat([nn_params, likelihood_params], axis=-1))

        # setup kernel and optimizer
        self.kernel = tfk.ExponentiatedQuadratic(length_scale=bandwidth)
        self.optim = tf.keras.optimizers.Adam(lr)

    def predict(self, x):
        # data handling
        x = self._handle_input_data(x, convert_to_tensor=True)
        x = self._normalize_data(x)

        # nn prediction
        nn_params, likelihood_std = self._split_into_nn_params_and_likelihood_std(self.particles)
        y_pred = self.nn.call_parametrized(x, nn_params)

        # form mixture of predictive distributions
        pred_dist = self.likelihood.get_pred_mixture_dist(y_pred, likelihood_std)

        # unnormalize preds
        y_pred = self._unnormalize_preds(y_pred)
        pred_dist = self._unnormalize_predictive_dist(pred_dist)
        return y_pred, pred_dist

    @tf.function
    def step(self, x_batch, y_batch):

        # compute posterior score (gradient of log prob)
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(self.particles)
            nn_params, likelihood_std = self._split_into_nn_params_and_likelihood_std(self.particles)

            # compute likelihood
            y_pred = self.nn.call_parametrized(x_batch, nn_params)  # (k, b, d)
            avg_log_likelihood = self.likelihood.log_prob(y_pred, y_batch, likelihood_std)

            if self.meta_learned_prior_mode:
                particles_reshaped = tf.reshape(self.particles, (self.prior.n_batched_priors,
                                                 self.n_particles // self.prior.n_batched_priors, -1))
                prior_prob = tf.reshape(self.prior.log_prob(particles_reshaped, model_params_prior_weight=self.prior_weight), (self.n_particles,))
            else:
                prior_prob = self.prior.log_prob(self.particles, model_params_prior_weight=self.prior_weight)

            # compute posterior log_prob
            prior_pre_factor = 1 / math.sqrt(self.num_train_samples) if self.sqrt_mode else 1 / self.num_train_samples
            post_log_prob = avg_log_likelihood + prior_pre_factor * prior_prob # (k,)
        score = tape.gradient(post_log_prob, self.particles)  # (k, n)

        # compute kernel matrix and grads
        particles_copy = tf.identity(self.particles)  # (k, n)
        with tf.GradientTape() as tape:
            tape.watch(self.particles)
            k_xx = self.kernel.matrix(self.particles, particles_copy)  # (k, k)
        k_grad = tape.gradient(k_xx, self.particles)
        svgd_grads_stacked = k_xx @ score - k_grad / self.n_particles  # (k, n)

        # apply SVGD gradients
        self.optim.apply_gradients([(- svgd_grads_stacked, self.particles)])
        return - post_log_prob


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

    nn = BayesianNeuralNetworkSVGD(x_train, y_train, hidden_layer_sizes=(64, 64), prior_weight=0.001, bandwidth=1000.0)

    n_iter_fit = 500
    for i in range(10):
        nn.fit(x_val=x_val, y_val=y_val, log_period=10, num_iter_fit=n_iter_fit)
        if d == 1:
            x_plot = tf.range(-8, 8, 0.1)
            nn.plot_predictions(x_plot, show=True)
