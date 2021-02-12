import math

import numpy as np
import tensorflow as tf

SEED = 245
SHUFFLING_RANDOM_SEED = 123
TAKS_SHUFFLING_RANDOM_SEED = 456

class MetaDatasetSampler:

    def __init__(self, data, per_task_batch_size, meta_batch_size=-1, n_batched_models=None, tiled=False,
                 flatten_data=False,  standardization=True, random_seed=SEED):
        """
        Encapsulates the meta-data with its statistics (mean and std) and provides batching functionality

        Args:
            data: list of tuples of ndarrays. Either [(train_x_1, train_t_1), ..., (train_x_n, train_t_n)] or
                    [(context_x, context_y, test_x, test_y), ( ... ), ...]
            per_task_batch_size (int): number of samples in each batch sampled from a task
            meta_batch_size: number of tasks sampled in each meta-training step
            n_batched_models: number of batched models for tiling. Is ignored when tiled = False
            tiled (bool): weather the returned batches should be tiled/stacked so that the first dimension
                        corresponds to the number of batched models
            standardization (bool): whether to compute standardization statistics. If False, the data statistics are
                                    set to zero mean and unit std --> standardization has no effect
        """
        self.n_tasks = len(data)

        tf.random.set_seed(random_seed)

        if meta_batch_size > self.n_tasks:
            print(f"NOTE: The requested meta batch size `{meta_batch_size}` is bigger the number "
                  f"of training tasks `{self.n_tasks}`. Reverting to using all of the tasks in each batch")
            meta_batch_size = -1

        if meta_batch_size == -1:
            meta_batch_size = self.n_tasks

        self.meta_batch_size = meta_batch_size
        self.tasks = []

        for task_data in data:
            if len(task_data) == 2:
                self.tasks.append(DatasetSampler(train_data=task_data, val_data=None,
                                                 batch_size=per_task_batch_size, n_batched_models=n_batched_models,
                                                 tiled=tiled, flatten_data=flatten_data))
            elif len(task_data) == 4:
                context_x, context_y, test_x, test_y = task_data
                train_data, val_data = (context_x, context_y), (test_x, test_y)
                self.tasks.append(DatasetSampler(train_data=train_data, val_data=val_data,
                                                 batch_size=per_task_batch_size, n_batched_models=n_batched_models,
                                                 tiled=tiled, flatten_data=flatten_data))
            else:
                raise Exception("Unexpected data shape")

        if per_task_batch_size == -1:
            # set to number of samples per task
            n_samples_per_task = [task_tuple[0].shape[0] for task_tuple in data]
            assert len(set(n_samples_per_task)) == 1, "n_samples differ across tasks --> per_task_batch_size must be set > 0"
            per_task_batch_size = n_samples_per_task[0]
        self.per_task_batch_size = per_task_batch_size

        self.input_dim = self.tasks[0].input_dim
        self.output_dim = self.tasks[0].output_dim
        self.n_train_samples = [task.n_train_samples for task in self.tasks]

        # Standardization of inputs and outputs
        self.standardization = standardization
        if standardization:
            self.x_mean, self.y_mean, self.x_std, self.y_std = self._compute_global_standardization_stats()
        else:
            self.x_mean, self.y_mean, self.x_std, self.y_std = self._get_zero_mean_unit_std_stats()

        self._update_task_standardization_stats()

        # Task batching
        tasks_ids = np.arange(self.n_tasks, dtype=np.int32)
        task_id_sampler = tf.data.Dataset.from_tensor_slices(tasks_ids).shuffle(self.n_tasks,
                                                                            seed=TAKS_SHUFFLING_RANDOM_SEED + 2,
                                                                            reshuffle_each_iteration=True).repeat().batch(1)
        self.task_id_sampler = task_id_sampler.__iter__()
        self.steps_per_epoch = math.ceil(self.n_tasks / self.meta_batch_size) * self.tasks[0].steps_per_epoch


    def get_meta_batch(self, shuffle=True):
        meta_batch_x, meta_batch_y = [], []
        n_train_samples, batch_size, task_ids = [], [], []

        for task_id in range(self.meta_batch_size):
            if shuffle:
                task_id = next(self.task_id_sampler)[0]

            task = self.tasks[task_id]

            # Get data sample
            x, y = task.get_batch()
            meta_batch_x.append(x)
            meta_batch_y.append(y)

            # Add additional data
            n_train_samples.append(task.n_train_samples)
            batch_size.append(task.batch_size)
            task_ids.append(task_id)

        meta_batch_x = tf.stack(meta_batch_x, axis=0)
        meta_batch_y = tf.stack(meta_batch_y, axis=0)

        n_train_samples = tf.convert_to_tensor(n_train_samples, tf.float32)
        batch_size = tf.convert_to_tensor(batch_size, tf.float32)
        task_ids = tf.convert_to_tensor(task_ids, tf.float32)

        return meta_batch_x, meta_batch_y, n_train_samples, batch_size, task_ids

    def copy_standardization_stats(self, obj):
        """
        Copies the standardization stats of an object to self
        """
        assert all([hasattr(obj, stats_var) for stats_var in ['x_mean', 'y_mean', 'x_std', 'y_std']])
        self.x_mean = obj.x_mean
        self.y_mean = obj.y_mean
        self.x_std = obj.x_std
        self.y_std = obj.y_std

    def plot_data(self, tasks_to_include, is_training, plot_val_data=True, ax=None):
        import matplotlib.pyplot as plt

        new_axis = False
        if ax is None:
            fig, (ax) = plt.subplots(1, 1, figsize=(4, 4))
            new_axis = True

        if tasks_to_include == None:
            tasks_to_include = range(self.n_tasks)

        for i in tasks_to_include:
            task = self.tasks[i]
            if is_training:
                task.plot_data(True, False, ax, f'Task {i+1}-', context=False)
            else:
                task.plot_data(True, plot_val_data, ax, f'Task {i+1}-')

        if new_axis:
            plt.legend()
            plt.show()

    def plot_prediction_functions(self, model, ax=None, plot_pred_std=False, plot_pred_lines=False, sample_functions=True, sample_from_prior=False, plot_data=False, title=None):
        assert(plot_pred_std or plot_pred_lines)
        import matplotlib.pyplot as plt
        x_min = tf.reduce_min([task.x_min for task in self.tasks])
        x_max = tf.reduce_max([task.x_max for task in self.tasks])

        flat_x = tf.linspace(x_min, x_max, 100)
        x = tf.reshape(flat_x, (100, self.output_dim))
        y_pred, pred_dist = model._predict(x, sample_functions=sample_functions, sample_from_prior=sample_from_prior)

        new_axis = False
        if ax is None:
            fig, (ax) = plt.subplots(1, 1, figsize=(4, 4))
            new_axis = True

        if plot_pred_std:
            std = tf.reshape(pred_dist.stddev(), (1,-1))[0]
            mean = tf.reshape(pred_dist.mean(), (1,-1))[0]
            top = mean + std
            bottom = mean - std

            ax.fill_between(flat_x, top, bottom, alpha=0.2)

        if plot_pred_lines:
            for i in range(y_pred.shape[0]):
                ax.plot(flat_x, tf.reshape(y_pred[i], (1, -1))[0], color='green', alpha=0.3, linewidth=1)

        if plot_data:
            self.plot_data(None, True, ax=ax)

        if new_axis:
            if title is None:
                ax.set_title("Prior functions sample" if sample_from_prior else "Posterior functions sample")
            else:
                ax.set_title(title)
            plt.show()

    def set_original_shape(self, original_shape):
        self.original_shape = original_shape

        # call setter in all tasks / dataset samplers
        for task in self.tasks:
            task.set_original_shape(original_shape)

    def _compute_global_standardization_stats(self):
        X_stack = tf.concat([task.train_data[0] for task in self.tasks], axis=0)
        Y_stack = tf.concat([task.train_data[1] for task in self.tasks], axis=0)

        x_mean = tf.math.reduce_mean(X_stack, axis=0, keepdims=True)
        y_mean = tf.math.reduce_mean(Y_stack, axis=0, keepdims=True)
        x_std = tf.math.reduce_std(X_stack, axis=0, keepdims=True) + 1e-8
        y_std = tf.math.reduce_std(Y_stack, axis=0, keepdims=True) + 1e-8

        tf.assert_equal(tf.rank(x_mean), 2)
        tf.assert_equal(tf.rank(y_mean), 2)
        tf.assert_equal(tf.rank(x_std), 2)
        tf.assert_equal(tf.rank(y_std), 2)

        return x_mean, y_mean, x_std, y_std

    def _get_zero_mean_unit_std_stats(self):
        x_mean = tf.zeros((1, self.input_dim), tf.float32)
        y_mean = tf.zeros((1, self.output_dim), tf.float32)
        x_std = tf.ones((1, self.input_dim), tf.float32)
        y_std = tf.ones((1, self.output_dim), tf.float32)
        return x_mean, y_mean, x_std, y_std

    def _update_task_standardization_stats(self):
        for task in self.tasks:
            task.copy_standardization_stats(self)

    def get_standardization_stats(self):
        return self.x_mean, self.y_mean, self.x_std, self.y_std

class DatasetSampler:
    def __init__(self, train_data, val_data, batch_size, n_batched_models=None, tiled=False, flatten_data=False,
                 x_mean=None, x_std=None, y_mean=None, y_std=None):
        self.train_data = self._handle_input_dimensionality(train_data[0], train_data[1])
        self._set_data_ranges_for_plotting()

        self.tiled = tiled
        self.flatten_data = flatten_data
        self.n_batched_models = n_batched_models
        self.n_train_samples = self.train_data[0].shape[0]
        assert not tiled or n_batched_models is not None

        if batch_size > self.n_train_samples:
            print(f"NOTE: The requested batch size `{batch_size}` is bigger the number of training samples `{self.n_train_samples}`")

        if batch_size == -1:
            batch_size = self.n_train_samples

        self.batch_size = batch_size
        self.steps_per_epoch = max(1, math.ceil(self.n_train_samples / self.batch_size))

        self.input_dim = self.train_data[0].shape[1]
        self.output_dim = self.train_data[1].shape[1]

        self._set_standardization_values(x_mean, x_std, y_mean, y_std)

        train_dataset = tf.data.Dataset.from_tensor_slices(self.train_data)

        train_dataset = train_dataset.shuffle(self.n_train_samples,
                                                    seed=SHUFFLING_RANDOM_SEED + 2,
                                                    reshuffle_each_iteration=True).repeat().batch(self.batch_size)

        self.train_batch_sampler = train_dataset.__iter__()

        if val_data is not None:
            self.val_data = self._handle_input_dimensionality(val_data[0], val_data[1])
            x_min_val, x_max_val = tf.math.floor(tf.reduce_min(self.val_data[0])), tf.math.ceil(tf.reduce_max(self.val_data[0]))

            x_min_val = tf.cast(x_min_val, tf.float32)
            x_max_val = tf.cast(x_max_val, tf.float32)

            self.x_min = tf.reduce_min([x_min_val, self.x_min])
            self.x_max = tf.reduce_max([x_max_val, self.x_max])

            self.n_val_samples = self.val_data[0].shape[0]

            val_dataset = tf.data.Dataset.from_tensor_slices(self.val_data)
            self.val_batch_sampler = val_dataset.shuffle(self.n_val_samples,
                                                         seed=SHUFFLING_RANDOM_SEED + 2,
                                                         reshuffle_each_iteration=True).batch(batch_size).__iter__()

    def get_batch(self):
        x, y = next(self.train_batch_sampler)
        x, y = self.process_batch(x, y)
        return x, y

    def process_batch(self, x, y):
        """
        Standardizes, reshapes and tiles both x and y if needed
        Args:
            x (tf.Tensor): input batch
            y (tf.Tensor): target batch

        Returns: processed (x, y)

        """
        x, y = self._standardize(x, y)
        x = self._reshape(x)

        if self.tiled:
            x, y = self._tile_batch(x, y)

        return x, y

    def process_eval_batch(self, x, y):
        """
        Standardizes, reshapes only x and tiles both x and y if needed
        Args:
            x (tf.Tensor): input batch
            y (tf.Tensor): target batch

        Returns: (x, y)
        """
        x, _ = self._standardize(x, None)
        x = self._reshape(x)

        if self.tiled:
            x, _ = self._tile_batch(x, y)

        return x, y

    def _tile_batch(self, x, y=None):
        tile_multiplies_x = tf.concat([tf.expand_dims(self.n_batched_models, 0), tf.ones_like(x.shape)], axis=0)
        x = tf.expand_dims(x, 0)
        x = tf.tile(x, tile_multiplies_x)

        if y is not None:
            tile_multiplies_y = tf.concat([tf.expand_dims(self.n_batched_models, 0), tf.ones_like(y.shape)], axis=0)
            y = tf.expand_dims(y, 0)
            y = tf.tile(y, tile_multiplies_y)

        return x, y

    def _reshape(self, x):
        """
        brings the data back into the original shape
        """
        if getattr(self, 'original_shape', None) is not None and not self.flatten_data:
            new_shape = tf.concat([x.shape[:-1], self.original_shape], axis=0)
            return tf.reshape(x, new_shape)
        return x

    def copy_standardization_stats(self, obj):
        """
        Copies the standardization stats of an object to self
        """
        assert all([hasattr(obj, stats_var) for stats_var in ['x_mean', 'y_mean', 'x_std', 'y_std']])
        self.x_mean = obj.x_mean
        self.y_mean = obj.y_mean
        self.x_std = obj.x_std
        self.y_std = obj.y_std

    def plot_data(self, train, val, ax=None, label_prefix="", context=True):
        import matplotlib.pyplot as plt
        assert (train or val)

        new_axis = False
        if ax is None:
            fig, (ax) = plt.subplots(1, 1, figsize=(4, 4))
            new_axis = True

        if val:
            x, y = self.val_data
            ax.scatter(x, y, label=f'{label_prefix}test', alpha=0.1)

        if train:
            x, y = self.train_data
            ax.scatter(x, y, color='black', label=f'{label_prefix}context' if context else f'{label_prefix}train', marker='x' if context else 'o', alpha=1 if context else 0.5)

        if new_axis:
            plt.legend()
            plt.show()

    def plot_prediction_functions(self, model, ax=None, plot_pred_std=False, plot_pred_lines=False, sample_functions=True, sample_from_prior=False, plot_data=True, title=None):
        assert(plot_pred_std or plot_pred_lines)
        import matplotlib.pyplot as plt
        flat_x = tf.linspace(self.x_min, self.x_max, self.n_val_samples)
        x = tf.reshape(flat_x, self.val_data[0].shape)
        y_pred, pred_dist = model._predict(x, sample_functions=sample_functions, sample_from_prior=sample_from_prior)

        new_axis = False
        if ax is None:
            fig, (ax) = plt.subplots(1, 1, figsize=(4, 4))
            new_axis = True

        if plot_pred_std:
            std = 2*tf.reshape(pred_dist.stddev(), (1, -1))[0]
            mean = tf.reshape(pred_dist.mean(), (1, -1))[0]
            top = mean + std
            bottom = mean - std

            ax.fill_between(flat_x, top, bottom, alpha=0.2)

        if plot_pred_lines:
            for i in range(y_pred.shape[0]):
                ax.plot(flat_x, tf.reshape(y_pred[i], (1, -1))[0], color='green', alpha=0.3, linewidth=1)

        if plot_data:
            self.plot_data(True, True, ax=ax)

        if new_axis:
            if title is None:
                ax.set_title("Prior functions sample" if sample_from_prior else "Posterior functions sample")
            else:
                ax.set_title(title)
            plt.show()

    @staticmethod
    def _handle_input_dimensionality(x, y):
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        y = tf.convert_to_tensor(y, dtype=tf.float32)

        if tf.rank(x) == 1:
            x = tf.expand_dims(x, -1)
        if tf.rank(y) == 1:
            y = tf.expand_dims(y, -1)

        tf.assert_equal(x.shape[0], y.shape[0])
        tf.assert_equal(tf.rank(x), 2)
        tf.assert_equal(tf.rank(y), 2)

        return x, y

    def _set_standardization_values(self, x_mean, x_std, y_mean, y_std):
        if x_mean is None:
            x_mean = tf.zeros(self.input_dim, tf.float32)
            y_mean = tf.zeros(self.output_dim, tf.float32)
            x_std = tf.ones(self.input_dim, tf.float32)
            y_std = tf.ones(self.output_dim, tf.float32)

        if tf.rank(x_mean) == 1:
            x_mean = tf.expand_dims(x_mean, 0)
            y_mean = tf.expand_dims(y_mean, 0)
            x_std = tf.expand_dims(x_std, 0)
            y_std = tf.expand_dims(y_std, 0)

        tf.assert_equal(tf.rank(x_mean), 2)
        tf.assert_equal(tf.rank(y_mean), 2)
        tf.assert_equal(tf.rank(x_std), 2)
        tf.assert_equal(tf.rank(y_std), 2)

        self.x_mean, self.y_mean = x_mean, y_mean
        self.x_std, self.y_std = x_std, y_std

    def _set_data_ranges_for_plotting(self):
        x_min, x_max = tf.math.floor(tf.reduce_min(self.train_data[0])), tf.math.ceil(tf.reduce_max(self.train_data[0]))
        self.x_min = tf.cast(x_min, dtype=tf.float32)
        self.x_max = tf.cast(x_max, dtype=tf.float32)

    def set_original_shape(self, original_shape):
        self.original_shape = original_shape

    def _standardize(self, x, y=None):
        x = tf.divide((x - self.x_mean), self.x_std)

        if y is not None:
            y = tf.divide((y - self.y_mean), self.y_std)

        return x, y


""" --- helper functions --- """

def _split_into_batches(list, max_batch_size):
    import math
    n_elements = len(list)
    if max_batch_size == -1:
        max_batch_size = n_elements
    n_batches = math.ceil(n_elements / float(max_batch_size))
    remainder = n_elements % n_batches
    batches = []
    idx = 0
    for i in range(n_batches):
        if i < remainder:
            batch_size = n_elements // n_batches + 1
        else:
            batch_size = n_elements // n_batches
        batches.append(list[idx:idx + batch_size])
        idx += batch_size

    assert all([len(batch) <= max_batch_size for batch in batches])
    assert np.sum([len(batch) for batch in batches]) == n_elements
    return batches
