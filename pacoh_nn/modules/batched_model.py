import tensorflow as tf


class TFModuleBatched(tf.keras.Model):
    def __init__(self, name, n_batched_models=None):
        super().__init__(name=name)
        self._variable_sizes = None
        self._parameters_shape = None
        self._n_batched_models = n_batched_models

    @tf.function
    def get_variables_vectorized(self):
        return tf.concat([tf.reshape(v, (1, -1)) for v in self.variables], axis=1)

    @tf.function
    def get_variables_stacked_per_model(self):
        return tf.reshape(self.get_variables_vectorized(), (self._n_batched_models, -1))

    @tf.function
    def variable_sizes(self):
        if self._variable_sizes is None:
            self._variable_sizes = [tf.size(v) for v in self.variables]
        return self._variable_sizes

    def copy_variables(self, obj):
        _vars = self.variables
        obj_vars = obj.variables

        for v1, v2 in zip(_vars, obj_vars):
            v1.assign(v2)

    @tf.function
    def set_variables_vectorized(self, parameters):
        if self._parameters_shape is None:
            self._parameters_shape = parameters.shape

        parameters = tf.reshape(parameters, (-1, 1))
        split = tf.split(parameters, self.variable_sizes())

        for v, n_v in zip(self.variables, split):
            v.assign(tf.reshape(n_v, v.shape))

    @tf.function
    def concat_and_vectorize_grads(self, gradients):
        vectorized_gradients = tf.concat([tf.reshape(g, (-1, 1)) for g in gradients], axis=0)
        if self._parameters_shape is None:
            return tf.reshape(vectorized_gradients, (self._n_batched_models, -1))

        return tf.reshape(vectorized_gradients, self._parameters_shape)

    @tf.function
    def split_and_unvectorize_grads(self, vectorized_grads):
        flat_grads = tf.reshape(vectorized_grads, (1, -1))
        parts = tf.split(flat_grads, self.variable_sizes(), axis=-1)
        return [tf.reshape(g, v.shape) for g, v in zip(parts, self.variables)]

    def call(self, x, **kwargs):
        raise NotImplementedError

    @tf.custom_gradient
    def call_parametrized(self, x, variables_vectorized):
        self.set_variables_vectorized(variables_vectorized)

        tape = tf.GradientTape(persistent=True)
        with tape:
            tape.watch([x] + list(self.trainable_variables))
            y = self.call(x, )

        def grad_fn(dy, variables):
            with tape:
                tampered_y = y * dy
            grads_x_w = tape.gradient(tampered_y, [x] + list(self.trainable_variables))
            grads_to_input = [grads_x_w[0], self.concat_and_vectorize_grads(grads_x_w[1:])]
            return grads_to_input, [None] * len(variables)

        return y, grad_fn