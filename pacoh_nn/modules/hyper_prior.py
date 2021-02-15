import tensorflow as tf
import tensorflow_probability as tfp

class GaussianHyperPrior(tf.Module):
    def __init__(self, batched_prior_module,
                 mean_mean=0.0, bias_mean_std=0.5, kernel_mean_std=0.5,
                 log_var_mean=-3.0, bias_log_var_std=0.5, kernel_log_var_std=0.5,
                 likelihood_log_var_mean_mean=-8,
                 likelihood_log_var_mean_std=1.0,
                 likelihood_log_var_log_var_mean=-4,
                 likelihood_log_var_log_var_std=0.2,
                 name='GaussianHyperPrior'):
        super().__init__(name=name)

        self.mean_mean = mean_mean
        self.bias_mean_std = bias_mean_std
        self.kernel_mean_std = kernel_mean_std

        self.log_var_mean = log_var_mean
        self.bias_log_var_std = bias_log_var_std
        self.kernel_log_var_std = kernel_log_var_std

        self.likelihood_log_var_mean_mean = likelihood_log_var_mean_mean
        self.likelihood_log_var_mean_std = likelihood_log_var_mean_std
        self.likelihood_log_var_log_var_mean = likelihood_log_var_log_var_mean
        self.likelihood_log_var_log_var_std = likelihood_log_var_log_var_std

        self.prior_module_names = [prior.name for prior in batched_prior_module.priors]
        self.n_batched_priors = len(self.prior_module_names)

        prior_module = batched_prior_module.priors[0]
        variables = prior_module.variables

        self.batched_variable_sizes = batched_prior_module.variable_sizes()
        self.batched_variable_names = [v.name for v in batched_prior_module.variables]

        self.base_variable_sizes = prior_module.base_variable_sizes
        self.base_variable_names = prior_module.base_variable_names

        for var in variables:
            self.process_variable(var)

    def process_variable(self, var):
        if 'model_parameters' in var.name:
            sizes = self.base_variable_sizes[0]
            names = self.base_variable_names[0]

            if 'mean' in var.name:
                # prior for model_parameters_mean
                means = []
                stds = []

                for v_size, v_name in zip(sizes, names):
                    mean = tf.ones((1, v_size), tf.float32) * self.mean_mean

                    if 'bias' in v_name:
                        std = tf.ones((1, v_size), tf.float32) * self.bias_mean_std
                    elif 'kernel' in v_name:
                        std = tf.ones((1, v_size), tf.float32) * self.kernel_mean_std
                    else:
                        raise Exception("Unexpected parameter")

                    means.append(mean)
                    stds.append(std)

                means = tf.concat(means, axis=1)
                means = tf.cast(tf.squeeze(means), tf.float32)
                stds = tf.concat(stds, axis=1)
                stds = tf.cast(tf.squeeze(stds), tf.float32)

                if tf.rank(means) == 0:
                    means = tf.expand_dims(means, 0)

                dist = tfp.distributions.Independent(tfp.distributions.Normal(means, stds), reinterpreted_batch_ndims=1)

                @tf.function
                def log_prob(parameters):
                    return dist.log_prob(parameters)

            elif 'log_var' in var.name:
                # prior for model_parameters_mean
                means = []
                stds = []

                for v_size, v_name in zip(sizes, names):
                    mean = tf.ones((1, v_size), tf.float32) * self.log_var_mean

                    if 'bias' in v_name:
                        std = tf.ones((1, v_size), tf.float32) * self.bias_log_var_std
                    elif 'kernel' in v_name:
                        std = tf.ones((1, v_size), tf.float32) * self.kernel_log_var_std
                    else:
                        raise Exception("Unexpected parameter")

                    means.append(mean)
                    stds.append(std)

                means = tf.concat(means, axis=1)
                means = tf.cast(tf.squeeze(means), tf.float32)
                stds = tf.concat(stds, axis=1)
                stds = tf.cast(tf.squeeze(stds), tf.float32)

                if tf.rank(means) == 0:
                    means = tf.expand_dims(means, 0)

                dist = tfp.distributions.Independent(
                    tfp.distributions.Normal(means, stds),
                    reinterpreted_batch_ndims=1)

                @tf.function
                def log_prob(parameters):
                    return dist.log_prob(parameters)
            else:
                raise Exception("Unexpected variable name")

        elif 'likelihood_parameters' in var.name:
            sizes = self.base_variable_sizes[1]

            if 'mean' in var.name:
                # prior for likelihood_parameters_mean
                means = tf.ones(sizes[0], tf.float32) * self.likelihood_log_var_mean_mean
                stds = tf.ones(sizes[0], tf.float32) * self.likelihood_log_var_mean_std

                means = tf.concat(means, axis=1)
                means = tf.cast(tf.squeeze(means), tf.float32)
                stds = tf.concat(stds, axis=1)
                stds = tf.cast(tf.squeeze(stds), tf.float32)

                if tf.rank(means) == 0:
                    means = tf.expand_dims(means, 0)

                dist = tfp.distributions.Independent(
                    tfp.distributions.Normal(means, stds),
                    reinterpreted_batch_ndims=1)

                @tf.function
                def log_prob(parameters):
                    return dist.log_prob(parameters)

            elif 'log_var' in var.name:
                # prior for likelihood_parameters_log_var
                means = tf.ones(sizes[0], tf.float32) * self.likelihood_log_var_log_var_mean
                stds = tf.ones(sizes[0], tf.float32) * self.likelihood_log_var_log_var_std
                means = tf.concat(means, axis=1)
                means = tf.cast(tf.squeeze(means), tf.float32)
                stds = tf.concat(stds, axis=1)
                stds = tf.cast(tf.squeeze(stds), tf.float32)

                if tf.rank(means) == 0:
                    means = tf.expand_dims(means, 0)

                dist = tfp.distributions.Independent(
                    tfp.distributions.Normal(means, stds),
                    reinterpreted_batch_ndims=1)

                @tf.function
                def log_prob(parameters):
                    return dist.log_prob(parameters)

            else:
                raise Exception("Unexpected variable name")
        else:
            raise Exception("Unexpeted variable name")

        suffix = var.name.split('/')[1]
        for prior_module_name in self.prior_module_names:
            name = f'{prior_module_name}/{suffix}'
            setattr(self, name, log_prob)

    @tf.function
    def log_prob_vectorized(self, params_vectorized, model_params_prior_weight=1.0):
        tf.assert_equal(params_vectorized.shape[1], self.n_batched_priors)
        parameters = tf.reshape(params_vectorized, (-1,))
        param_split = tf.split(parameters, self.batched_variable_sizes)
        log_probs = tf.reshape([getattr(self, v_name)(v) for v, v_name in zip(param_split, self.batched_variable_names)],
                               (self.n_batched_priors, -1))
        prefactor = tf.reshape([model_params_prior_weight if 'model' in var_name else 1.0 for var_name in self.batched_variable_names],
                   (self.n_batched_priors, -1))
        log_probs *= prefactor
        log_probs = tf.reduce_sum(log_probs, axis=-1)
        return log_probs

    def log_prob(self, variables):
        log_probs = tf.reshape([getattr(self, v.name)(v) for v in variables], (self.n_batched_priors, -1))
        log_probs = tf.reduce_sum(log_probs, axis=-1, keepdims=True)
        return log_probs


