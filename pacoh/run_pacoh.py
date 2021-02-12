import tensorflow as tf
import numpy as np

import math

from datasets.regression_datasets import MetaDataset

class SinusoidDataset(MetaDataset):

    def __init__(self, amp_low=0.7, amp_high=1.3,
                 period_low=1.5, period_high=1.5,
                 x_shift_mean=0.0, x_shift_std=0.1,
                 y_shift_mean=5.0, y_shift_std=0.1,
                 slope_mean=0.5, slope_std=0.2,
                 noise_std=0.1, x_low=-5, x_high=5, random_state=None):

        super().__init__(random_state)
        assert y_shift_std >= 0 and noise_std >= 0, "std must be non-negative"
        self.amp_low, self.amp_high = amp_low, amp_high
        self.period_low, self.period_high = period_low, period_high
        self.y_shift_mean, self.y_shift_std = y_shift_mean, y_shift_std
        self.x_shift_mean, self.x_shift_std = x_shift_mean, x_shift_std
        self.slope_mean, self.slope_std = slope_mean, slope_std
        self.noise_std = noise_std
        self.x_low, self.x_high = x_low, x_high

    def generate_meta_test_data(self, n_tasks, n_samples_context, n_samples_test):
        assert n_samples_test > 0
        meta_test_tuples = []
        for i in range(n_tasks):
            f = self._sample_sinusoid()
            X = self.random_state.uniform(self.x_low, self.x_high, size=(n_samples_context + n_samples_test, 1))
            Y = f(X) + self.noise_std * self.random_state.normal(size=f(X).shape)
            meta_test_tuples.append(
                (X[:n_samples_context], Y[:n_samples_context], X[n_samples_context:], Y[n_samples_context:]))

        return meta_test_tuples

    def generate_meta_train_data(self, n_tasks, n_samples):
        meta_train_tuples = []
        for i in range(n_tasks):
            f = self._sample_sinusoid()
            X = self.random_state.uniform(self.x_low, self.x_high, size=(n_samples, 1))
            Y = f(X) + self.noise_std * self.random_state.normal(size=f(X).shape)
            meta_train_tuples.append((X, Y))
        return meta_train_tuples

    def _sample_sinusoid(self):
        amplitude = self.random_state.uniform(self.amp_low, self.amp_high)
        x_shift = self.random_state.normal(loc=self.x_shift_mean, scale=self.x_shift_std)
        y_shift = self.random_state.normal(loc=self.y_shift_mean, scale=self.y_shift_std)
        slope = self.random_state.normal(loc=self.slope_mean, scale=self.slope_std)
        period = self.random_state.uniform(self.period_low, self.period_high)
        return lambda x: slope * x + amplitude * np.sin(period * (x - x_shift)) + y_shift


def generate_meta_data():
    from matplotlib import pyplot as plt
    env = SinusoidDataset(amp_low=1.0, amp_high=1.0, slope_mean=0, slope_std=0.0, x_shift_mean=0.0, x_shift_std=1.0)

    meta_train_data = env.generate_meta_train_data(n_tasks=100, n_samples=20)
    meta_val_data = env.generate_meta_test_data(n_tasks=20, n_samples_context=20, n_samples_test=200)
    # for x, y in meta_train_data:
    #     plt.scatter(x, y)
    # plt.show()

    return meta_train_data, meta_val_data



def main():
    from matplotlib import pyplot as plt
    # setup data set
    from datasets.regression_datasets import provide_data
    #meta_train_data, meta_val_data, _ = provide_data(dataset='sin', n_train_tasks=20, n_samples=5)
    meta_train_data, meta_test_data = generate_meta_data()

    from pacoh.pacoh_nn_regression import PACOH_NN_Regression
    pacoh_model = PACOH_NN_Regression(meta_train_data, random_seed=22, num_iter_meta_train=200,
                                         learn_likelihood=True, likelihood_std=tf.exp(0.5 * -4.631251503547343))

    pacoh_model.plot_prior(plot_pred_lines=True, plot_pred_std=True, plot_data=True, max_task_to_plot=10, show=True)

    pacoh_model.meta_fit(meta_test_data[:10], eval_period=20000, log_period=1000, plot_prior_during_training=True)


    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.0))
    for i in range(2):
        x_context, y_context, x_test, y_test = meta_test_data[i]

        # plotting
        x_plot = tf.range(-5, 5, 0.1)
        pacoh_model.plot_posterior(x_context, y_context, x_plot, ax=axes[i])
        axes[i].scatter(x_test, y_test, color='blue', alpha=0.2, label="test data")
        axes[i].scatter(x_context, y_context, color='red', label="train data")
        axes[i].legend()
        axes[i].set_xlabel('x')
        axes[i].set_xlabel('y')


if __name__ == '__main__':
    main()