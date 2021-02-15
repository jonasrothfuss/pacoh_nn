import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors


class AffineTransform:
    def __init__(self, normalization_mean, normalization_std):

        self.loc_tensor = tf.cast(normalization_mean, dtype=tf.float32)
        self.scale_tensor = tf.cast(normalization_std, dtype=tf.float32)

        shift = tfb.Shift(self.loc_tensor)

        if tf.size(self.scale_tensor) == 1:
            scale = tfb.Scale(self.scale_tensor)
        else:
            scale = tfb.ScaleMatvecDiag(self.scale_tensor)

        self.transform = shift(scale)

    def apply(self, base_dist):
        if isinstance(base_dist, tfp.distributions.Categorical):
            # Categorical distribution --> make sure that normalization stats are mean=0 and std=1
            tf.assert_equal(tf.math.count_nonzero(self.loc_tensor), tf.constant(0, dtype=tf.int64))
            tf.assert_equal(tf.math.count_nonzero(self.scale_tensor - 1.0), tf.constant(0, dtype=tf.int64))
            return base_dist
        else:
            base_dist = base_dist

            d = self.transform(base_dist)
            d.transform = self.transform
            d.base_dist = base_dist

            def cdf(y, **kwargs):
                x = self.transform.inverse(y)
                return base_dist.cdf(x, **kwargs)

            def mean():
                return self.transform(base_dist.mean())

            def stddev():
                return tf.math.exp(tf.math.log(base_dist.stddev()) + tf.math.log(self.scale_tensor))

            def variance():
                return tf.math.exp(tf.math.log(base_dist.variance()) + 2 * tf.math.log(self.scale_tensor))

            d.cdf = cdf
            d.mean = mean
            d.stddev = stddev
            d.variance = variance

            return d
