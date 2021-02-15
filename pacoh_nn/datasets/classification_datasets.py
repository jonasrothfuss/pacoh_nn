import numpy as np
import os
import glob
from collections import OrderedDict

# Data directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OMNIGLOT_DIR = os.path.join(DATA_DIR, 'omniglot_resized')
FMNIST_DIR = os

# Random seed setting
TRAIN_VAL_TEST_CLASS_SAMPLER_RANDOM_SEED = 123


class MetaDataset:
    def __init__(self, random_state=None, original_shape=None):
        if random_state is None:
            self.random_state = np.random
        else:
            self.random_state = random_state

        self.original_shape = original_shape

    def generate_meta_train_data(self, n_tasks: int, n_samples: int) -> list:
        raise NotImplementedError

    def generate_meta_test_data(self, n_tasks: int, n_samples_context: int, n_samples_test: int) -> list:
        raise NotImplementedError

class OmniglotMetaDataset(MetaDataset):
    img_height = 28
    img_width = 28
    img_depth = 1
    dim_x = img_height * img_width
    num_alphabets_test = 20
    num_alphabets_train = 30
    max_num_classes = 14

    def __init__(self, n_classes=5, data_folder=OMNIGLOT_DIR, ignore_alphabet_boundaries=False, random_state=None):
        """
        Omniglot dataset consisting of characters of various alphabets. The first time this class is instantiated, the
        dataset is downloaded to the specified data folder.

        Args:
            n_classes (int): number of classes (--> makes it k-way classification problem)
            data_folder (str): path where data is (to be) stored
            random_state (np.random.RandomState): random number generator state
        """
        super().__init__(random_state=random_state, original_shape=self.get_img_attributes())

        assert n_classes <= self.max_num_classes

        self.data_folder = data_folder
        self.n_classes = n_classes
        self.max_n_tasks_per_alphabet = self.max_num_classes // self.n_classes
        self.ignore_alphabet_boundaries = ignore_alphabet_boundaries

        # A) make sure that data is downloaded
        if not self._dataset_downloaded():
            self._download_dataset()

        # B) get character folders for the tasks
        self.class_folders_per_alphabet = self._get_class_folders_per_alphabet(shuffle=True)

    def generate_meta_train_data(self, n_tasks=-1, n_samples=-1):
        """
        Generates meta training data (image classification tasks)

        Args:
            n_tasks (int): number of tasks
            n_samples (int): number of samples (images) per task

        Returns:
            List of 2-tuples [(x_train_1, y_train_1), ... (x_train_n, y_train_n)] wherein x_train is a stack
            of flattened images with shape (n_samples, 784) and y_train a stack of one-hot encodings with shape
            (n_samples, n_classes)
        """
        class_folders_per_alphabet = self._get_alphabet_split("train", self.class_folders_per_alphabet)

        # check number of tasks
        max_num_tasks = self.max_n_tasks_per_alphabet * self.num_alphabets_train
        n_tasks = n_tasks if n_tasks > 0 else max_num_tasks
        assert self.ignore_alphabet_boundaries or n_tasks <= max_num_tasks

        # generate tasks and convert corresponding data to numpy arrays
        class_folders_per_task = self._split_alphabets_in_tasks(class_folders_per_alphabet, n_tasks)
        task_tuples = self._select_and_load_samples_to_numpy(class_folders_per_task, n_samples_context=n_samples,
                                                             n_samples_test=0)

        # make final checks
        assert len(task_tuples) == n_tasks
        assert all([x_train.shape[0] == y_train.shape[0] for x_train, y_train in task_tuples])

        return task_tuples

    def generate_meta_test_data(self, n_tasks=-1, n_samples_context=10, n_samples_test=-1):
        """
           Generates meta test data (image classification tasks)

           Args:
               n_tasks (int): number of tasks
               n_samples_context (int): number of context samples (images) per task
               n_samples_test (int): number of test samples per task

           Returns:
               List of n_tasks 4-tuples [(x_context_1, y_context_1, x_test_1, y_test_1), ... ]
        """
        class_folders_per_alphabet = self._get_alphabet_split("test", self.class_folders_per_alphabet)

        # check number of tasks
        max_num_tasks = self.max_n_tasks_per_alphabet * self.num_alphabets_test
        n_tasks = n_tasks if n_tasks > 0 else max_num_tasks
        assert self.ignore_alphabet_boundaries or n_tasks <= max_num_tasks

        # generate tasks and convert corresponding data to numpy arrays
        class_folders_per_task = self._split_alphabets_in_tasks(class_folders_per_alphabet, n_tasks)
        task_tuples = self._select_and_load_samples_to_numpy(class_folders_per_task, n_samples_context, n_samples_test)

        # make final checks
        assert len(task_tuples) == n_tasks
        assert all([x_context.shape[1] == x_test.shape[1] for x_context, _, x_test, _ in task_tuples])
        assert all([y_context.shape[1] == y_test.shape[1] == self.n_classes for _, y_context, _, y_test in task_tuples])

        return task_tuples

    def get_img_attributes(self):
        return self.img_height, self.img_width, self.img_depth

    def _select_and_load_samples_to_numpy(self, class_folders_per_task, n_samples_context, n_samples_test):

        task_tuples = []

        for class_folders in class_folders_per_task.values():
            context_path_labels, test_path_labels = self._select_samples_per_task(class_folders, n_samples_context, n_samples_test)

            # load images and convert label to one hot encoding
            x_context = np.stack([load_img_to_numpy(img_path) for img_path, _ in context_path_labels])
            y_context = np.stack([one_hot(label, n_classes=self.n_classes) for _, label in context_path_labels])

            assert x_context.shape[0] == y_context.shape[0]

            if n_samples_test == 0:
                task_tuples.append((x_context, y_context))
            else:
                x_test = np.stack([load_img_to_numpy(img_path) for img_path, _ in test_path_labels])
                y_test = np.stack([one_hot(label, n_classes=self.n_classes) for _, label in test_path_labels])
                assert x_test.shape[0] == y_test.shape[0]
                task_tuples.append((x_context, y_context, x_test, y_test))

        return task_tuples

    def _select_samples_per_task(self, class_folders, n_samples_context, n_samples_test):

        assert self.n_classes == len(class_folders)
        assert not (n_samples_context < 0 and n_samples_test < 0)

        context_image_paths, test_image_paths = [], []
        context_labels, test_labels = [], []

        img_paths_per_class = OrderedDict([(class_id, glob.glob(os.path.join(class_folder_path, '*.png')))
                                        for class_id, class_folder_path in enumerate(class_folders)])
        n_images = np.sum([len(img_paths) for img_paths in img_paths_per_class.values()])

        if n_samples_context < 0:
            n_samples_context = n_images - n_samples_test
        if n_samples_test < 0:
            assert n_samples_context > 0
            n_samples_test = n_images - n_samples_context

        n_samples = n_samples_test + n_samples_context
        assert n_samples <= n_images

        assert n_samples_context % self.n_classes == 0, 'n_samples must be a multiple of the number of classes'
        assert n_samples_test % self.n_classes == 0, 'n_samples must be a multiple of the number of classes'

        n_samples_context_per_class = n_samples_context // self.n_classes
        n_samples_test_per_class = n_samples_test // self.n_classes
        n_samples_per_class = n_samples_context_per_class + n_samples_test_per_class

        for class_id, img_paths in img_paths_per_class.items():
            img_paths = self.random_state.choice(img_paths, n_samples_per_class, replace=False)

            if n_samples_test != 0:
                test_image_paths.extend(img_paths[n_samples_context_per_class:])
                test_labels.extend([class_id for _ in range(n_samples_per_class - n_samples_context_per_class)])

            context_image_paths.extend(img_paths[:n_samples_context_per_class])
            context_labels.extend([class_id for _ in range(n_samples_context_per_class)])

        context_path_labels = list(zip(context_image_paths, context_labels))
        test_path_labels = list(zip(test_image_paths, test_labels))

        assert len(context_path_labels) == n_samples_context
        assert len(test_path_labels) == n_samples_test
        # check that there is no overlap between context and test
        assert n_samples_test == 0 or self.ignore_alphabet_boundaries or\
               len(set(list(zip(*context_path_labels))[0]).intersection(set(list(zip(*test_path_labels))[0]))) == 0

        return context_path_labels, test_path_labels

    def _get_class_folders_per_alphabet(self, shuffle=False):
        alphabet_folders = glob.glob(os.path.join(self.data_folder, '*/*'))

        tasks = []
        for alphabet_folder in alphabet_folders:
            task_name = os.path.basename(alphabet_folder)
            character_folders = glob.glob(os.path.join(alphabet_folder, '*'))
            tasks.append((task_name, character_folders))
        class_folders_per_task = sorted(tasks, key=lambda x: x[0])

        if shuffle:
            np.random.RandomState(TRAIN_VAL_TEST_CLASS_SAMPLER_RANDOM_SEED).shuffle(class_folders_per_task)

        return class_folders_per_task

    def _get_alphabet_split(self, dataset_type, class_folders_per_alphabet):
        if dataset_type == 'train':
            return class_folders_per_alphabet[:self.num_alphabets_train]
        elif dataset_type == 'test':
            return class_folders_per_alphabet[self.num_alphabets_train:]
        else:
            raise Exception("Invalid dataset type")

    def _split_alphabets_in_tasks(self, class_folders_per_alphabet, n_tasks):
        """
        splits alphabets into K-way classification tasks
        """
        import math
        if self.ignore_alphabet_boundaries:
            import itertools
            class_folders = list(
                itertools.chain(*[class_folders for alphabet, class_folders in class_folders_per_alphabet]))
            class_folders_per_task = []
            for task_id in range(n_tasks):
                class_folders_per_task.append((task_id, list(self.random_state.choice(class_folders, size=self.n_classes))))
            class_folders_per_task = OrderedDict(class_folders_per_task)
        else:
            n_tasks_per_alphabet = math.ceil(n_tasks / len(class_folders_per_alphabet))
            assert n_tasks_per_alphabet <= self.max_n_tasks_per_alphabet

            class_folders_per_task = []
            for alphabet_name, class_folders in class_folders_per_alphabet:
                self.random_state.shuffle(sorted(class_folders))
                for i in range(n_tasks_per_alphabet):
                    class_folders_for_task = class_folders[i * self.n_classes:(i + 1) * self.n_classes]
                    task_name = alphabet_name + '_%i' % i
                    class_folders_per_task.append((task_name, class_folders_for_task))
            assert len(class_folders_per_task) == len(class_folders_per_alphabet) * n_tasks_per_alphabet

            # choose a subset of n_tasks
            selected_indices = self.random_state.choice(len(class_folders_per_task), n_tasks, replace=False)
            class_folders_per_task = OrderedDict([class_folders_per_task[i] for i in selected_indices])
        assert len(class_folders_per_task) == n_tasks
        return class_folders_per_task

    def _download_dataset(self) -> None:
        """
        Downloads the omniglot dataset and resizes the images to 28 x 28
        """
        URL_IMG_BACKGROUND = 'https://github.com/brendenlake/omniglot/raw/master/python/images_background.zip'
        URL_IMG_EVAL = 'https://github.com/brendenlake/omniglot/raw/master/python/images_evaluation.zip'

        print("Downloading Omniglot dataset to %s"%self.data_folder)
        download_and_extract(URL_IMG_BACKGROUND, self.data_folder)
        download_and_extract(URL_IMG_EVAL, self.data_folder)

        print("Resizing omniglot images")
        resize_omniglot()

    def _dataset_downloaded(self) -> bool:
        """
        checks whether the omniglot has been dowloaded and can be found in the expected data directory

        Returns:
            boolean that indicates whether the dataset has is available under the expected location
        """
        all_images = glob.glob(os.path.join(self.data_folder, '*/*/*/*.png'))
        number_of_images = len(all_images)
        return number_of_images > 32400

class FMNISTToyMetaDataset(MetaDataset):
    img_height = 28
    img_width = 28
    img_depth = 1
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    n_classes = 10

    def __init__(self, random_state=None):
        super().__init__(random_state=random_state, original_shape=self.get_img_attributes())
        self._load_and_process_data()

    def generate_meta_train_data(self, n_tasks=10, n_samples=100):
        """
        Generates meta training data (image classification tasks)

        Args:
            n_tasks (int): number of tasks
            n_samples (int): number of samples (images) per task

        Returns:
            List of 2-tuples [(x_train_1, y_train_1), ... (x_train_n, y_train_n)] wherein x_train is a stack
            of flattened images with shape (n_samples, 784) and y_train a stack of one-hot encodings with shape
            (n_samples, n_classes)
        """
        assert n_tasks > 0 or n_samples > 0

        if n_tasks < 0:
            n_tasks = self.n_train_images // n_samples
        if n_samples < 0:
            n_samples = self.n_train_images // n_tasks

        assert n_samples * n_tasks <= self.n_train_images

        indices = self.random_state.choice(self.n_train_images, n_samples * n_tasks, replace=False)

        task_tuples = [(self.train_images[task_indices], self.train_labels[task_indices])
                       for task_indices in np.split(indices, n_tasks)]

        assert len(task_tuples) == n_tasks
        assert all([x_train.shape[0] == y_train.shape[0] == n_samples for x_train, y_train in task_tuples])

        return task_tuples

    def generate_meta_test_data(self, n_tasks=10, n_samples_context=100, n_samples_test=500):
        """
           Generates meta test data (image classification tasks)

           Args:
               n_tasks (int): number of tasks
               n_samples_context (int): number of context samples (images) per task
               n_samples_test (int): number of test samples per task

           Returns:
               List of n_tasks 4-tuples [(x_context_1, y_context_1, x_test_1, y_test_1), ... ]
        """
        assert n_tasks > 0 and n_samples_context > 0 and n_samples_test > 0
        n_samples = n_samples_context + n_samples_test
        assert n_tasks * n_samples <= self.n_test_images

        indices = self.random_state.choice(self.n_test_images, n_samples * n_tasks, replace=False)

        task_tuples = []
        for task_idx in np.split(indices, n_tasks):
            context_idx = task_idx[:n_samples_context]
            test_idx = task_idx[n_samples_context:]
            x_context, y_context = self.test_images[context_idx], self.test_labels[context_idx]
            x_test, y_test = self.test_images[test_idx], self.test_labels[test_idx]
            task_tuples.append((x_context, y_context, x_test, y_test))

        # make final checks
        assert len(task_tuples) == n_tasks
        assert all([x_context.shape[1] == x_test.shape[1] for x_context, _, x_test, _ in task_tuples])
        assert all([y_context.shape[1] == y_test.shape[1] == self.n_classes for _, y_context, _, y_test in task_tuples])

        return task_tuples

    def get_img_attributes(self):
        return self.img_height, self.img_width, self.img_depth

    def _load_and_process_data(self):
        from tensorflow import keras
        import tensorflow as tf
        fashion_mnist = keras.datasets.fashion_mnist
        (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

        train_images = train_images / 255.0
        test_images = test_images / 255.0

        self.train_images = tf.reshape(train_images, (train_images.shape[0], -1)).numpy()
        self.train_labels = tf.one_hot(train_labels, 10).numpy()

        self.test_images = tf.reshape(test_images, (test_images.shape[0], -1)).numpy()
        self.test_labels = tf.one_hot(test_labels, 10).numpy()

        self.n_train_images = self.train_images.shape[0]
        self.n_test_images = self.test_images.shape[0]

""" ---- helper functions ---- """

def download_and_extract(url, target_dir):
    import requests, zipfile, io
    r = requests.get(url)
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        z.extractall(target_dir)

def resize_omniglot():
    from PIL import Image

    image_path = os.path.join(OMNIGLOT_DIR, '*/*/*/')
    all_images = glob.glob(image_path + '*')

    for i, image_file in enumerate(all_images):
        im = Image.open(image_file)
        im = im.resize((28, 28), resample=Image.LANCZOS)
        im.save(image_file)
        if i % 2000 == 0:
            print('resized %i out of 32400 images'%i)

def load_img_to_numpy(img_path):
    import imageio
    img = imageio.imread(img_path)
    img = 1.0 - img.astype(np.float32) / 255.0  # convert to float, scale and invert
    if img.ndim == 2:
        img = img.flatten()
    elif img.ndim == 3:
        img = img.reshape((-1, img.shape[-1]))
    return img

def one_hot(i, n_classes):
    one_hot = np.zeros(n_classes, dtype=np.float32)
    one_hot[i] = 1.0
    return one_hot

def plot_images(images, labels, n_images=100, plot_size=None):
    if plot_size is None:
        plot_size = (26, 20)

    import matplotlib.pyplot as plt
    n = int(np.ceil(np.sqrt(n_images)))
    fig, axes = plt.subplots(n, n, figsize=plot_size)
    axes = axes.flatten()
    for label, img, ax in zip(labels[:n_images], images[:n_images], axes):
        ax.imshow(np.reshape(img, (28, 28)))
        ax.set_title(label)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    """ --- Omniglot --- """
    dataset = OmniglotMetaDataset()
    meta_train_data = dataset.generate_meta_train_data()
    meta_test_data = dataset.generate_meta_test_data()

    task_data = meta_train_data[0]
    plot_images(*task_data)

    task_data = meta_train_data[3][:2]
    plot_images(*task_data)

    """ --- Fashion MNIST --- """
    dataset = FMNISTToyMetaDataset()
    meta_train_data = dataset.generate_meta_train_data()

    task_data = meta_train_data[3][:2]
    plot_images(*task_data)

    meta_test_data = dataset.generate_meta_test_data()
    task_data = meta_test_data[3][:2]
    plot_images(*task_data)