import tensorflow as tf

from modules.batched_model import TFModuleBatched


class FullyConnectedNN(TFModuleBatched):
    def __init__(self, output_size, hidden_layer_sizes=(32, 32), activation='relu', name='nn'):
        super().__init__(name=name)
        self.n_hidden_layers = len(hidden_layer_sizes)
        for i, n_units in enumerate(hidden_layer_sizes):
            setattr(self, 'layer_%i' % (i + 1), tf.keras.layers.Dense(n_units, activation=activation))
        self.layer_out = tf.keras.layers.Dense(output_size, activation=None)

    def call(self, x, **kwargs):
        for i in range(self.n_hidden_layers):
            x = getattr(self, 'layer_%i' % (i + 1))(x)
        return self.layer_out(x)


class BatchedFullyConnectedNN(TFModuleBatched):
    def __init__(self, n_batched_models, output_size, hidden_layer_sizes=(32, 32), activation='relu',
                 name='batched_nn'):
        super().__init__(name=name, n_batched_models=n_batched_models)
        self.n_batched_models = n_batched_models

        self.models = []
        for i in range(n_batched_models):
            self.models.append(FullyConnectedNN(output_size=output_size,
                                                hidden_layer_sizes=hidden_layer_sizes,
                                                activation=activation,
                                                name=f'{self.name}_{i}'))

    def call(self, inputs, batched_input=False):
        if batched_input:
            # inputs: (n_batched_models, batch_size, input_shape)
            tf.assert_equal(len(inputs.shape), 3)
            tf.assert_equal(inputs.shape[0], self.n_batched_models)
            outputs = tf.stack([self.models[i](inputs[i]) for i in range(self.n_batched_models)])
        else:
            # inputs: (batch_size, input_shape)
            tf.assert_equal(len(inputs.shape), 2)
            outputs = tf.stack([self.models[i](inputs) for i in range(self.n_batched_models)])
        return outputs  # (n_batched_models, batch_size, output_shape)
