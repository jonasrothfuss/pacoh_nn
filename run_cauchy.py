import tensorflow as tf

def main():
    tf.get_logger().setLevel('ERROR')

    from pacoh_nn.datasets.regression_datasets import provide_data

    meta_train_data, meta_test_data, _ = provide_data(dataset='cauchy_20')

    from pacoh_nn.pacoh_nn_regression import PACOH_NN_Regression
    pacoh_model = PACOH_NN_Regression(meta_train_data, prior_weight=0.184, bandwidth=480.,
                                      hyper_prior_likelihood_log_var_mean_mean=-1.0,
                                      hyper_prior_log_var_mean=-1.74, hyper_prior_nn_std=0.12,
                                      hyper_prior_weight=1e-5, lr=1.5e-3, learn_likelihood=True,
                                      random_seed=28, num_iter_meta_train=20000)

    pacoh_model.meta_fit(meta_test_data[:10], eval_period=10000, log_period=1000, plot_period=5000)

    eval_metrics_mean, eval_metrics_std = pacoh_model.meta_eval_datasets(meta_test_data)
    for key in eval_metrics_mean:
        print("%s: %.4f +- %.4f" % (key, eval_metrics_mean[key], eval_metrics_std[key]))


if __name__ == '__main__':
    main()