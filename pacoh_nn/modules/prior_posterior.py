import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd

#from pacoh_old.modules.batched_module import TFModuleBatched
from pacoh_nn.modules import TFModuleBatched


class GaussianPriorPerVariable(tf.keras.Model):

    def __init__(self, models_for_init, add_init_noise=True, init_noise_std=0.1,
                 likelihood_param_size=0, likelihood_prior_mean=tf.math.log(0.01), likelihood_prior_std=1.0,
                 name='gaussian_prior'):
        """
        Gaussian prior for a NN model
        Args:
            model: NN model or list of NN models
            likelihood: likelihood object
            config (dict): configuration dict
            name (str): module name
        """
        super().__init__(name=name)

        self.init_noise_std = init_noise_std

        self.base_variable_sizes = []
        self.base_variable_names = []

        # initialize prior associated with the model parameters
        assert type(models_for_init) == list
        self._init_model_prior_from_models(models=models_for_init, init_noise_std=init_noise_std)

        # initialize prior associated with the likelihood parameters
        self._init_likelihood_prior(likelihood_param_size, likelihood_prior_mean, likelihood_prior_std)

        # size of the model vs. likelihood parameters so that they can be separated if necessary
        self.split_sizes = [tf.reduce_sum(v_s).numpy() for v_s in self.base_variable_sizes]

        # add noise to initialization
        if add_init_noise:
            self._add_noise_to_variables(init_noise_std)

    @tf.function
    def sample(self, n_samples, use_init_std=False):
        """
        Generates n_samples samples from the prior distribution
        Args:
            n_samples (int): number of samples to draw
            use_init_std (bool): whether to use the init std

        Returns: Tensor of shape (n_samples, parameter_size)

        """
        mp = self._sample_param('model_parameters', n_samples, use_init_std)

        if self.learn_likelihood_variables:
            lp = self._sample_param('likelihood_parameters', n_samples, use_init_std)
            return tf.concat([mp, lp], axis=1)
        else:
            return mp

    @tf.function
    def log_prob(self, parameters, model_params_prior_weight=1.0):
        split = tf.split(parameters, self.split_sizes, axis=-1)

        log_prob = model_params_prior_weight * self._log_prob('model_parameters', split[0])

        if self.learn_likelihood_variables:
            log_prob += self._log_prob('likelihood_parameters', split[1])

        return log_prob

    def _sample_param(self, param_name, n_samples, use_init_std):
        param_name = f'{self.name}/{param_name}'

        name = f'{param_name}_mean'
        mean = getattr(self, name)

        name = f'{param_name}_log_var'
        log_var = getattr(self, name)

        if use_init_std:
            std = tf.ones_like(log_var) * self.config['init_std']
        else:
            std = tf.math.exp(0.5 * log_var)

        dist = tfp.distributions.Independent(
            tfp.distributions.Normal(mean, std),
            reinterpreted_batch_ndims=1)

        return dist.sample(n_samples)

    def _log_prob(self, param_name, param):
        param_name = f'{self.name}/{param_name}'

        name = f'{param_name}_mean'
        mean = getattr(self, name)

        name = f'{param_name}_log_var'
        log_var = getattr(self, name)
        std = tf.math.exp(0.5 * log_var)

        dist = tfp.distributions.Independent(
            tfp.distributions.Normal(mean, std),
            reinterpreted_batch_ndims=1)

        return dist.log_prob(param)

    def _init_model_prior_from_models(self, models, init_noise_std=0.1):
        # check that model variables have the same size
        assert len(set([tuple(model.get_variables_vectorized().shape) for model in models])) == 1

        # save variable names and sizes
        self.base_variable_names.append([v.name for v in models[0].variables])
        self.base_variable_sizes.append(models[0].variable_sizes())

        means = []
        log_vars = []

        for variable_tuple in zip(*[model.variables for model in models]):
            # assert that the variables have the same shape and name
            assert len(set([tuple(var.shape) for var in variable_tuple])) == 1
            assert len(set([var.name.split('/')[-1] for var in variable_tuple]))

            var_name = variable_tuple[0].name
            var_shape = variable_tuple[0].shape

            var_stack = tf.stack(variable_tuple, axis=0)
            init_std = tf.math.reduce_std(var_stack, axis=0)  # std of model initializations

            if 'kernel' in var_name:
                # take the initialization of the first model as mean of the prior
                init_mean = var_stack[0]
                # take half of the std across model initializations as std of the prior
                init_log_var = 2 * tf.math.log(0.5 * (init_std + 1e-8))
            elif 'bias' in var_name:
                # sample prior mean of bias
                init_mean = tf.random.normal(var_shape, mean=0.0, stddev=init_noise_std)
                # use init_std for the log_var
                init_log_var = tf.ones(var_shape) * 2 * tf.math.log(0.5 * (init_noise_std + 1e-8))
            else:
                raise Exception("Unknown variable type")


            means.append(tf.reshape(init_mean, (-1,)))
            log_vars.append(tf.reshape(init_log_var, (-1,)))

        means = tf.concat(means, axis=0)
        log_vars = tf.concat(log_vars, axis=0)

        name = f'{self.name}/model_parameters_mean'
        setattr(self, name, tf.Variable(tf.squeeze(means), dtype=tf.float32, name=name, trainable=True))

        name = f'{self.name}/model_parameters_log_var'
        setattr(self, name, tf.Variable(tf.squeeze(log_vars), dtype=tf.float32, name=name, trainable=True))

    def _init_likelihood_prior(self, likelihood_param_size, likelihood_prior_mean, likelihood_prior_std):
        self.learn_likelihood_variables = False
        if likelihood_param_size > 0:
                self.learn_likelihood_variables = True
                self.base_variable_names.append(['std'])
                self.base_variable_sizes.append([likelihood_param_size])

                mean = tf.ones(likelihood_param_size, tf.float32) * likelihood_prior_mean
                name = f'{self.name}/likelihood_parameters_mean'
                setattr(self, name, tf.Variable(mean, dtype=tf.float32, name=name, trainable=True))

                log_var = tf.ones(likelihood_param_size, tf.float32) * tf.math.log(likelihood_prior_std)
                name = f'{self.name}/likelihood_parameters_log_var'
                setattr(self, name, tf.Variable(log_var, dtype=tf.float32, name=name, trainable=True))

    def _add_noise_to_variables(self, init_noise_std):
        # exclude mean and log_var for model parameters, since they have been initialized from the models
        # --> they are already noisy
        vars_to_add_noise = [var for var in self.variables if not 'model_parameters' in var.name]
        for v in vars_to_add_noise:
            noise = tf.random.normal(v.shape, mean=0.0, stddev=init_noise_std)
            v.assign_add(noise)

class BatchedGaussianPrior(TFModuleBatched):

    def __init__(self, batched_model, n_batched_priors, likelihood_param_size=0, likelihood_prior_mean=tf.math.log(0.1),
                 likelihood_prior_std=1.0, add_init_noise=True, init_noise_std=0.1, name='batched_Gaussian_prior'):
        """
        Batched Gaussian priors for the model
        Args:
            batched_model: a batched NN model for which to instantiate a prior
            likelihood: likelihood object
            config (dict): configuration for the prior
            name (str): name of the module
        """
        super().__init__(name=name, n_batched_models=n_batched_priors)

        self.n_batched_priors = n_batched_priors
        self.priors = []

        n_models = len(batched_model.models)
        self.n_models_per_prior = n_models // self.n_batched_priors
        assert self.n_models_per_prior > 0

        for i in range(self.n_batched_priors):
            models_for_init = batched_model.models[i*self.n_models_per_prior:(i+1)*self.n_models_per_prior]
            self.priors.append(GaussianPriorPerVariable(models_for_init=models_for_init, add_init_noise=add_init_noise,
                                                        init_noise_std=init_noise_std,
                                                        likelihood_param_size=likelihood_param_size,
                                                        likelihood_prior_mean=likelihood_prior_mean,
                                                        likelihood_prior_std=likelihood_prior_std,
                                                        name=f'{self.name}_{i}'))

        self.split_sizes = self.priors[0].split_sizes
        self._variable_sizes = None

    @tf.function
    def log_prob(self, parameters, model_params_prior_weight=1.0):
        # parameters should have dimensions [#priors, #samples_per_prior, #params]
        log_prob = tf.stack([self.priors[i].log_prob(parameters[i], model_params_prior_weight=model_params_prior_weight)
                             for i in range(self.n_batched_priors)])
        return log_prob

    def sample(self, n_samples, use_init_std=False):
        sample = tf.stack([self.priors[i].sample(n_samples, use_init_std)
                           for i in range(self.n_batched_priors)])
        return sample

    @tf.custom_gradient
    def sample_parametrized(self, n_samples, variables_vectorized):
        self.set_variables_vectorized(variables_vectorized)

        tape = tf.GradientTape(persistent=True)
        with tape:
            tape.watch(list(self.trainable_variables))
            samples = self.sample(n_samples)

        def grad_fn(dy, variables):
            with tape:
                tampered_samples = samples * dy
            grads_n_w = tape.gradient(tampered_samples, list(self.trainable_variables))
            grads_to_input = [None, self.concat_and_vectorize_grads(grads_n_w)]
            return grads_to_input, [None] * len(variables)

        return samples, grad_fn


class GaussianPosterior(tf.Module):
    def __init__(self, stacked_nn_init_params, likelihood_param_size=0):
        super().__init__()
        self.likelihood_param_size = likelihood_param_size

        # mean & std for nn params
        nn_param_size = stacked_nn_init_params.shape[-1]
        mean_nn_params = tf.zeros(nn_param_size)
        post_init_std = tfp.stats.stddev(stacked_nn_init_params, sample_axis=0) + 1.0 / nn_param_size
        log_stddev_nn_params = post_init_std

        # mean & std for likelihood params
        mean_likelihood_params = -2 * tf.ones(likelihood_param_size)
        log_stddev_likelihood_params = tf.ones(likelihood_param_size)

        self.mean = tf.Variable(tf.concat([mean_nn_params, mean_likelihood_params], axis=0))
        self.log_std = tf.Variable(tf.math.log(tf.concat([log_stddev_nn_params, log_stddev_likelihood_params], axis=0)))

    @property
    def stddev(self):
        return tf.math.exp(self.log_stddev)

    @property
    def dist(self):
        return tfd.Independent(tfd.Normal(self.mean, tf.exp(self.log_std)), reinterpreted_batch_ndims=1)

    def sample(self, size):
        return self.dist.sample(size)

    def log_prob(self, param_values):
        return self.dist.log_prob(param_values)


class GaussianPrior(tf.Module):

    def __init__(self, nn_param_size, nn_prior_std, likelihood_param_size=0,
                 likelihood_prior_mean=tf.math.log(0.1), likelihood_prior_std=1.0):
        super().__init__()

        self.nn_param_size = nn_param_size
        self.likelihood_param_size = likelihood_param_size

        nn_prior_mean = tf.zeros(nn_param_size)
        nn_prior_std = tf.ones(nn_param_size) * nn_prior_std
        self.prior_dist_nn = tfd.Independent(tfd.Normal(nn_prior_mean, nn_prior_std), reinterpreted_batch_ndims=1)

        # mean and std of the Normal distribution over the log_std of the likelihood
        likelihood_prior_mean = tf.ones(likelihood_param_size) * likelihood_prior_mean
        likelihood_prior_std = tf.ones(likelihood_param_size) * likelihood_prior_std
        self.prior_dist_likelihood = tfd.Independent(tfd.Normal(likelihood_prior_mean, likelihood_prior_std),
                                                     reinterpreted_batch_ndims=1)

    def sample(self, size):
        return tf.concat([self.prior_dist_nn.sample(size), self.prior_dist_likelihood.sample(size)], axis=-1)

    def log_prob(self, param_values, model_params_prior_weight=1.0):
        nn_params, likelihood_params = self._split_params(param_values)
        log_prob_nn = self.prior_dist_nn.log_prob(nn_params)
        log_prob_likelihood = self.prior_dist_likelihood.log_prob(likelihood_params)
        return model_params_prior_weight * log_prob_nn + log_prob_likelihood

    def _split_params(self, params):
        assert params.shape[-1] == self.nn_param_size + self.likelihood_param_size
        nn_params, likelihood_params = tf.split(params, [self.nn_param_size, self.likelihood_param_size], axis=-1)
        return nn_params, likelihood_params


if __name__ == '__main__':
    stacked_nn_init_params = tf.random.uniform([7, 10])
    likelihood_param_size = 1
    posterior = GaussianPosterior(stacked_nn_init_params, likelihood_param_size)
    for i in posterior.trainable_variables:
        print(i.shape)
