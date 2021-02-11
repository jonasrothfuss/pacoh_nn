import tensorflow as tf


class RBF_Kernel(tf.Module):
    def __init__(self, bandwidth):
        super().__init__()
        self.bandwidth = bandwidth

    def __call__(self, X, Y):
        dnorm2 = norm_sq_tf(X, Y)
        gamma = 1.0 / (1e-8 + 2 * self.bandwidth ** 2)
        K_XY = tf.exp(-gamma * dnorm2)
        return K_XY


def norm_sq_tf(X, Y):
    XX = tf.matmul(X, X, transpose_b=True)
    XX = tf.expand_dims(tf.linalg.diag_part(XX, k=1), -1)
    XY = tf.matmul(X, Y, transpose_b=True)
    YY = tf.matmul(Y, Y, transpose_b=True)
    YY = tf.expand_dims(tf.linalg.diag_part(YY, k=1), -2)
    return -2 * XY + XX + YY