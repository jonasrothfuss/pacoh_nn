import numpy as np
import pandas as pd
from scipy.stats import truncnorm
import os
import h5py
import yaml
import copy

X_LOW = -5
X_HIGH = 5

Y_HIGH = 2.5
Y_LOW = -2.5

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data')

PHYSIONET_URL = 'https://www.dropbox.com/sh/2ocmumrq3o9ak8u/AAAIHPW7XZj_5uCX0gkaQZRLa?dl=1'
PHYSIONET_DIR = os.path.join(DATA_DIR, 'physionet2012')

SWISSFEL_URL = 'https://www.dropbox.com/sh/rabvj5jyw3lb7zh/AABcLeFmGTOMtjeTzqG81FUZa?dl=1'
SWISSFEL_DIR = os.path.join(DATA_DIR, 'swissfel')

BERKELEY_SENSOR_URL = 'https://www.dropbox.com/sh/y6egx20lod1gsrs/AACyXAk9Ua7SI-q1tpEb1SHba?dl=1'
BERKELEY_SENSOR_DIR = os.path.join(DATA_DIR, 'sensor_data')

MHC_DATA_URL = 'https://www.dropbox.com/sh/1hd9ey7pj7vhfw9/AADKgv5B1rLMBaYDrarbIyiea?dl=1'
MHC_DATA_DIR = os.path.join(DATA_DIR, 'mhc_data')

class MetaDataset:
    def __init__(self, random_state=None):
        if random_state is None:
            self.random_state = np.random
        else:
            self.random_state = random_state

    def generate_meta_train_data(self, n_tasks: int, n_samples: int) -> list:
        raise NotImplementedError

    def generate_meta_test_data(self, n_tasks: int, n_samples_context: int, n_samples_test: int) -> list:
        raise NotImplementedError


class PhysionetMetaDataset(MetaDataset):

    def __init__(self, random_state=None, variable_id=0, dtype=np.float32, physionet_dir=None):
        super().__init__(random_state)
        self.dtype = dtype
        if physionet_dir is not None:
            self.data_dir = physionet_dir
        elif PHYSIONET_DIR is not None:
            self.data_dir = PHYSIONET_DIR
        else:
            raise ValueError("No data directory provided.")

        if not os.path.isdir(self.data_dir):
            print("Physionet data does not exist in %s" % self.data_dir)
            download_and_unzip_data(PHYSIONET_URL, self.data_dir)

        self.variable_list = ['GCS', 'Urine', 'HCT', 'BUN', 'Creatinine', 'DiasABP']

        assert variable_id < len(self.variable_list), "Unknown variable ID"
        self.variable = self.variable_list[variable_id]

        self.data_path = os.path.join(self.data_dir, "set_a_merged.h5")

        with pd.HDFStore(self.data_path, mode='r') as hdf_file:
            self.keys = hdf_file.keys()

    def generate_meta_train_data(self, n_tasks, n_samples=47):
        """
        Samples n_tasks patients and returns measurements from the variable
        with the ID variable_id. n_samples defines in this case the cut-off
        of hours on the ICU, e.g., n_samples=24 returns all measurements that
        were taken in the first 24 hours. Generally, those will be less than
        24 measurements. If there are less than n_tasks patients that have
        any measurements of variable variable_id before hour n_samples, the
        returned list will contain less than n_tasks tuples.
        """

        assert n_tasks <= 500, "We don't have that many tasks"
        assert n_samples < 48, "We don't have that many samples"

        meta_train_tuples = []

        for patient in self.keys:
            df = pd.read_hdf(self.data_path, patient, mode='r')[self.variable].dropna()
            times = df.index.values.astype(self.dtype)
            values = df.values.astype(self.dtype)
            times_context = [time for time in times if time <= n_samples]
            if len(times_context) > 0:
                times_context = np.array(times_context, dtype=self.dtype)
                values_context = values[:len(times_context)]
                if values_context.shape[0] >= 4:
                    meta_train_tuples.append((times_context, values_context))
                else:
                    continue
            if len(meta_train_tuples) >= n_tasks:
                break

        return meta_train_tuples

    def generate_meta_test_data(self, n_tasks, n_samples_context=24,
                                n_samples_test=-1, variable_id=0):
        """
        Samples n_tasks patients and returns measurements from the variable
        with the ID variable_id. n_samples defines in this case the cut-off
        of hours on the ICU, e.g., n_samples=24 returns all measurements that
        were taken in the first 24 hours. Generally, those will be less than
        24 measurements. The remaining measurements are returned as test points,
        i.e., n_samples_test is unused.
        If there are less than n_tasks patients that have any measurements
        of variable variable_id before hour n_samples, the
        returned list will contain less than n_tasks tuples.
        """

        assert n_tasks <= 1000, "We don't have that many tasks"
        assert n_samples_context < 48, "We don't have that many samples"

        meta_test_tuples = []

        for patient in reversed(self.keys):
            df = pd.read_hdf(self.data_path, patient, mode='r')[self.variable].dropna()
            times = df.index.values.astype(self.dtype)
            values = df.values.astype(self.dtype)
            times_context = [time for time in times if time <= n_samples_context]
            times_test = [time for time in times if time > n_samples_context]
            if len(times_context) > 0 and len(times_test) > 0:
                times_context = np.array(times_context, dtype=self.dtype)
                times_test = np.array(times_test, dtype=self.dtype)
                values_context = values[:len(times_context)]
                values_test = values[len(times_context):]
                if values_context.shape[0] >= 4:
                    meta_test_tuples.append((times_context, values_context,
                                             times_test, values_test))
                else:
                    continue
            if len(meta_test_tuples) >= n_tasks:
                break

        return meta_test_tuples

class SinusoidMetaDataset(MetaDataset):

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

class GPFunctionsMetaDataset(MetaDataset):

    def __init__(self, noise_std=0.1, lengthscale=1.0, mean=0.0, x_low=-5, x_high=5, random_state=None):
        self.noise_std, self.lengthscale, self.mean = noise_std, lengthscale, mean
        self.x_low, self.x_high = x_low, x_high
        super().__init__(random_state)

    def generate_meta_test_data(self, n_tasks, n_samples_context, n_samples_test):
        assert n_samples_test > 0
        meta_test_tuples = []
        for i in range(n_tasks):
            X = self.random_state.uniform(self.x_low, self.x_high, size=(n_samples_context + n_samples_test, 1))
            Y = self._gp_fun_from_prior(X)
            meta_test_tuples.append(
                (X[:n_samples_context], Y[:n_samples_context], X[n_samples_context:], Y[n_samples_context:]))

        return meta_test_tuples

    def generate_meta_train_data(self, n_tasks, n_samples):
        meta_train_tuples = []
        for i in range(n_tasks):
            X = self.random_state.uniform(self.x_low, self.x_high, size=(n_samples, 1))
            Y = self._gp_fun_from_prior(X)
            meta_train_tuples.append((X, Y))
        return meta_train_tuples

    def _gp_fun_from_prior(self, X):
        assert X.ndim == 2

        n = X.shape[0]

        def kernel(a, b, lengthscale):
            sqdist = np.sum(a ** 2, 1).reshape(-1, 1) + np.sum(b ** 2, 1) - 2 * np.dot(a, b.T)
            return np.exp(-.5 * (1 / lengthscale) * sqdist)

        K_ss = kernel(X, X, self.lengthscale)
        L = np.linalg.cholesky(K_ss + 1e-8 * np.eye(n))
        f = self.mean + np.dot(L, self.random_state.normal(size=(n, 1)))
        y = f + self.random_state.normal(scale=self.noise_std, size=f.shape)
        return y

class CauchyMetaDataset(MetaDataset):

    def __init__(self, noise_std=0.05, ndim_x=2, random_state=None):
        self.noise_std = noise_std
        self.ndim_x = ndim_x
        super().__init__(random_state)

    def generate_meta_train_data(self, n_tasks, n_samples):
        meta_train_tuples = []
        for i in range(n_tasks):
            X = truncnorm.rvs(-3, 2, loc=0, scale=2.5, size=(n_samples, self.ndim_x), random_state=self.random_state)
            Y = self._gp_fun_from_prior(X)
            meta_train_tuples.append((X, Y))
        return meta_train_tuples

    def generate_meta_test_data(self, n_tasks, n_samples_context, n_samples_test):
        assert n_samples_test > 0
        meta_test_tuples = []
        for i in range(n_tasks):
            X = truncnorm.rvs(-3, 2, loc=0, scale=2.5, size=(n_samples_context + n_samples_test, self.ndim_x),
                              random_state=self.random_state)
            Y = self._gp_fun_from_prior(X)
            meta_test_tuples.append(
                (X[:n_samples_context], Y[:n_samples_context], X[n_samples_context:], Y[n_samples_context:]))

        return meta_test_tuples

    def _mean(self, x):
        loc1 = -1 * np.ones(x.shape[-1])
        loc2 = 2 * np.ones(x.shape[-1])
        cauchy1 = 1 / (np.pi * (1 + (np.linalg.norm(x - loc1, axis=-1)) ** 2))
        cauchy2 = 1 / (np.pi * (1 + (np.linalg.norm(x - loc2, axis=-1)) ** 2))
        return 6 * cauchy1 + 3 * cauchy2 + 1

    def _gp_fun_from_prior(self, X):
        assert X.ndim == 2

        n = X.shape[0]

        def kernel(a, b, lengthscale):
            sqdist = np.sum(a ** 2, 1).reshape(-1, 1) + np.sum(b ** 2, 1) - 2 * np.dot(a, b.T)
            return np.exp(-.5 * (1 / lengthscale) * sqdist)

        K_ss = kernel(X, X, 0.5)
        L = np.linalg.cholesky(K_ss + 1e-8 * np.eye(n))
        f = self._mean(X) + np.dot(L, self.random_state.normal(scale=0.2, size=(n, 1))).flatten()
        y = f + self.random_state.normal(scale=self.noise_std, size=f.shape)
        return y.reshape(-1, 1)


""" Swissfel Dataset"""

class SwissfelMetaDataset(MetaDataset):
    runs_12dim = [
        {'experiment': '2018_10_31/line_ucb_ascent', 'run': 0},
        {'experiment': '2018_10_31/line_ucb_ascent', 'run': 1},
        {'experiment': '2018_10_31/line_ucb_ascent', 'run': 2},
        {'experiment': '2018_10_31/line_ucb', 'run': 0},
        {'experiment': '2018_10_31/line_ucb', 'run': 1},
        {'experiment': '2018_10_31/line_ucb', 'run': 2},
        {'experiment': '2018_10_31/neldermead', 'run': 0},
        {'experiment': '2018_10_31/neldermead', 'run': 1},
        {'experiment': '2018_10_31/neldermead', 'run': 2},
    ]

    runs_24dim = [
        {'experiment': '2018_11_01/line_ucb_ascent_bpm_24', 'run': 0},
        {'experiment': '2018_11_01/line_ucb_ascent_bpm_24', 'run': 1},
        {'experiment': '2018_11_01/line_ucb_ascent_bpm_24', 'run': 3},
        {'experiment': '2018_11_01/line_ucb_ascent_bpm_24_small', 'run': 0},
        {'experiment': '2018_11_01/lipschitz_line_ucb_bpm_24', 'run': 0},
        {'experiment': '2018_11_01/neldermead_bpm_24', 'run': 0},
        {'experiment': '2018_11_01/neldermead_bpm_24', 'run': 1},
        {'experiment': '2018_11_01/parameter_scan_bpm_24', 'run': 0},
    ]

    def __init__(self, random_state=None, param_space_id=0, swissfel_dir=None):
        super().__init__(random_state)

        self.swissfel_dir = SWISSFEL_DIR if swissfel_dir is None else swissfel_dir

        if not os.path.isdir(self.swissfel_dir):
            print("Swissfel data does not exist in %s"%self.swissfel_dir)
            download_and_unzip_data(SWISSFEL_URL, self.swissfel_dir)


        if param_space_id == 0:
            run_specs = copy.deepcopy(self.runs_12dim)
        elif param_space_id == 1:
            run_specs = copy.deepcopy(self.runs_24dim)
        else:
            raise NotImplementedError

        self.random_state.shuffle(run_specs)
        self.run_specs_train = run_specs[:5]
        self.run_specs_test = run_specs[5:]

    def _load_data(self, experiment, run=0):
        path = os.path.join(self.swissfel_dir, experiment)

        # read hdf5
        hdf5_path = os.path.join(path, 'data/evaluations.hdf5')
        dset = h5py.File(hdf5_path, 'r')
        run = str(run)
        data = dset['1'][run][()]
        dset.close()

        # read config and recover parameter names

        config_path = os.path.join(path, 'experiment.yaml')
        config_file = open(config_path, 'r')  # 'document.yaml' contains a single YAML document.

        # get config files for parameters
        files = yaml.load(config_file)['swissfel.interface']['channel_config_set']
        if not isinstance(files, list):
            files = [files]

        files += ['channel_config_set.txt']  # backwards compatibility

        parameters = []
        for file in files:
            params_path = os.path.join(path, 'sf', os.path.split(file)[1])
            if not os.path.exists(params_path):
                continue

            frame = pd.read_csv(params_path, comment='#')

            parameters += frame['pv'].tolist()

        return data, parameters

    def _load_meta_dataset(self, train=True):
        run_specs = self.run_specs_train if train else self.run_specs_test
        data_tuples = []
        for run_spec in run_specs:
            data, parameters = self._load_data(**run_spec)
            data_tuples.append((data['x'], data['y']))

        assert len(set([X.shape[-1] for X, _ in data_tuples])) == 1
        assert all([X.shape[0] == Y.shape[0] for X, Y in data_tuples])
        return data_tuples

    def generate_meta_train_data(self, n_tasks=5, n_samples=200):
        assert n_tasks == len(self.run_specs_train), "number of tasks must be %i" % len(self.run_specs_train)
        meta_train_tuples = self._load_meta_dataset(train=True)

        max_n_samples = max([X.shape[0] for X, _ in meta_train_tuples])
        assert n_samples <= max_n_samples, 'only %i number of samples available' % max_n_samples

        meta_train_tuples = [(X[:n_samples], Y[:n_samples]) for X, Y in meta_train_tuples]

        return meta_train_tuples

    def generate_meta_test_data(self, n_tasks=None, n_samples_context=200, n_samples_test=400):
        if n_tasks is None:
            n_tasks = len(self.run_specs_test)

        assert n_tasks == len(self.run_specs_test), "number of tasks must be %i" % len(self.run_specs_test)
        meta_test_tuples = self._load_meta_dataset(train=False)

        max_n_samples = min([X.shape[0] for X, _ in meta_test_tuples])
        assert n_samples_context + n_samples_test <= max_n_samples, 'only %i number of samples available' % max_n_samples

        idx = np.arange(n_samples_context + n_samples_test)
        self.random_state.shuffle(idx)
        idx_context, idx_test = idx[:n_samples_context], idx[n_samples_context:]

        meta_test_tuples = [(X[idx_context], Y[idx_context], X[idx_test], Y[idx_test]) for X, Y in meta_test_tuples]

        return meta_test_tuples

""" MHC complex """

class MHCMetaDataset(MetaDataset):

    def __init__(self, cv_split_id=0, random_state=None):
        super().__init__(random_state)
        assert cv_split_id in [0,1]
        if cv_split_id == 0:
            self.test_task_ids = [0, 6]
        else:
            self.test_task_ids = [3, 5]
        self.train_task_ids = sorted(list(set(range(7)) - set(self.test_task_ids)))

        if not os.path.isdir(MHC_DATA_DIR):
            print("MHC-I data does not exist in %s"%MHC_DATA_DIR)
            download_and_unzip_data(MHC_DATA_URL, MHC_DATA_DIR)


    def generate_meta_test_data(self, n_tasks=2, n_samples_context=20, n_samples_test=-1):
        assert n_tasks <= 2
        task_tuples = self._load_data()
        test_tuples = []
        for task_id in self.test_task_ids[:n_tasks]:
            x, y = task_tuples[task_id]
            x_context, y_context = x[:n_samples_context], y[:n_samples_context]
            if n_samples_test == -1:
                x_test, y_test = x[n_samples_context:], y[n_samples_context:]
            else:
                x_test = x[n_samples_context:n_samples_context+n_samples_test],
                y_test = y[n_samples_context:n_samples_context+n_samples_test]
            test_tuples.append((x_context, y_context, x_test, y_test))
        return test_tuples

    def generate_meta_train_data(self, n_tasks=5, n_samples=-1):
        assert n_tasks <= 5
        task_tuples = self._load_data()
        train_tuples = []
        for task_id in self.train_task_ids[:n_tasks]:
            x, y = task_tuples[task_id]
            train_tuples.append((x[:n_samples], y[:n_samples]))
        return train_tuples

    def _load_data(self):
        from scipy.io import loadmat
        data_path = os.path.join(MHC_DATA_DIR, 'iedb_benchmark.mat')
        data = loadmat(data_path)

        self.MHC_alleles = data['MHC_alleles']
        task_tuples = []
        for task_id in range(7):
            idx_task = np.where(data['contexts'].flatten() == task_id)
            x = data['examples'].T[idx_task]
            y = data['labels'][idx_task]
            task_tuples.append((x, y))
        return task_tuples

""" Berkeley Sensor data """

class BerkeleySensorMetaDataset(MetaDataset):

    def __init__(self, random_state=None, separate_train_test_days=True):
        super().__init__(random_state)
        task_ids = np.arange(46)
        self.random_state.shuffle(task_ids)
        self.train_task_ids = task_ids[:36]
        self.test_task_ids = task_ids[36:]
        self.separate_train_test_days = separate_train_test_days # whether to also seperate the meta-train and meta-test set by days

        if not os.path.isdir(BERKELEY_SENSOR_DIR):
            print("Berkeley-Sensor data does not exist in %s" % BERKELEY_SENSOR_DIR)
            download_and_unzip_data(BERKELEY_SENSOR_URL, BERKELEY_SENSOR_DIR)

    def generate_meta_test_data(self, n_tasks=10, n_samples_context=144, n_samples_test=-1):
        task_tuples = self._load_data()

        if n_samples_test == -1:
            n_samples_test = min(2*self.n_points_per_day, 3*self.n_points_per_day - n_samples_context)
        else:
            assert n_samples_context + n_samples_test <= 3*self.n_points_per_day

        test_tuples = []
        for task_id in self.test_task_ids[:n_tasks]:
            x, y = task_tuples[task_id]
            start_idx = -1 * (n_samples_test + n_samples_context)
            x_context, y_context = x[start_idx:-n_samples_test], y[start_idx:-n_samples_test]
            x_test, y_test = x[-n_samples_test:], y[-n_samples_test:]
            test_tuples.append((x_context, y_context, x_test, y_test))
        return test_tuples

    def generate_meta_train_data(self, n_tasks=36, n_samples=-1):
        task_tuples = self._load_data()
        if self.separate_train_test_days:
            if n_samples == -1:
                n_samples = 2*self.n_points_per_day
            else:
                assert n_samples <= 2*self.n_points_per_day
        train_tuples = []
        for task_id in self.train_task_ids[:n_tasks]:
            x, y = task_tuples[task_id]
            train_tuples.append((x[:n_samples], y[:n_samples]))
        return train_tuples

    def _load_data(self, lags=10):
        from scipy.io import loadmat

        data_path = os.path.join(BERKELEY_SENSOR_DIR, 'berkeley_data.mat')
        data = loadmat(data_path)['berkeley_data']['data'][0][0]
        # replace outlier
        data[4278, 6] = (data[4278 - 1, 6] + data[4278 + 1, 6]) / 2
        n_points_per_day_raw = int(data.shape[0] / 5)
        daytime = np.concatenate([np.arange(n_points_per_day_raw) / n_points_per_day_raw for _ in range(5)])

        # remove first day since it has a break with the remaining 3 days (i.e. day 1, 5, 6, 7, 8]
        data = data[n_points_per_day_raw:]
        daytime = daytime[n_points_per_day_raw:]

        data_tuples = []
        for i in range(data.shape[-1]):
            time_series = data[:, i]
            y = time_series[lags:]
            x = np.stack([time_series[lag: -lags + lag] for lag in range(lags)] + [daytime[lags:]], axis=-1)
            assert x.shape[0] == y.shape[0] == len(time_series) - lags
            # subsample every 5 minutes
            x = x[::10]
            y = y[::10]

            data_tuples.append((x,y))

        self.n_points_per_day = int(data_tuples[0][0].shape[0] / 4)
        return data_tuples


def download_and_unzip_data(url, target_dir):
    from urllib.request import urlopen
    from zipfile import ZipFile
    print('Downloading %s'%url)
    tempfilepath = os.path.join(DATA_DIR, 'tempfile.zip')
    zipresp = urlopen(url)
    with open(tempfilepath, 'wb') as f:
        f.write(zipresp.read())
    zf = ZipFile(tempfilepath)
    print('Extracting to %s' % target_dir)
    zf.extractall(path=target_dir)
    zf.close()
    os.remove(tempfilepath)



""" Data provider """

def provide_data(dataset, seed=28, n_train_tasks=None, n_samples=None, config=None):
    import numpy as np

    N_TEST_TASKS = 20
    N_VALID_TASKS = 20
    N_TEST_SAMPLES = 200

    # if specified, overwrite default settings
    if config is not None:
        if config['num_test_valid_tasks'] is not None: N_TEST_TASKS = config['num_test_valid_tasks']
        if config['num_test_valid_tasks'] is not None: N_VALID_TASKS = config['num_test_valid_tasks']
        if config['num_test_valid_samples'] is not None:  N_TEST_SAMPLES = config['num_test_valid_samples']

    """ Prepare Data """
    if 'sin-nonstat' in dataset:
        if len(dataset.split('_')) == 2:
            n_train_tasks = int(dataset.split('_')[-1])

        dataset = SinusoidNonstationaryMetaDataset(random_state=np.random.RandomState(seed + 1))

        if n_samples is None:
            n_train_samples = n_context_samples = 20
        else:
            n_train_samples = n_context_samples = n_samples

        if n_train_tasks is None: n_train_tasks = 20

    elif 'sin' in dataset:
        if len(dataset.split('_')) == 2:
            n_train_tasks = int(dataset.split('_')[-1])

        dataset = SinusoidMetaDataset(random_state=np.random.RandomState(seed + 1))

        if n_samples is None:
            n_train_samples = n_context_samples = 5
        else:
            n_train_samples = n_context_samples = n_samples

        if n_train_tasks is None: n_train_tasks = 20

    elif 'gp_funcs' in dataset:
        dataset = GPFunctionsMetaDataset(random_state=np.random.RandomState(seed + 1))

        if n_samples is None:
            n_train_samples = n_context_samples = 5
        else:
            n_train_samples = n_context_samples = n_samples

        if n_train_tasks is None: n_train_tasks = 20

    elif 'cauchy' in dataset:
        if len(dataset.split('_')) == 2:
            n_train_tasks = int(dataset.split('_')[-1])

        dataset = CauchyMetaDataset(random_state=np.random.RandomState(seed + 1))

        if n_samples is None:
            n_train_samples = n_context_samples = 20
        else:
            n_train_samples = n_context_samples = n_samples

        if n_train_tasks is None: n_train_tasks = 20


    elif 'physionet' in dataset:
        variable_id = int(dataset[-1])
        assert 0 <= variable_id <= 5
        dataset = PhysionetMetaDataset(random_state=np.random.RandomState(seed + 1), variable_id=variable_id)
        n_context_samples = 24
        n_train_samples = 47

        n_train_tasks = 100
        # N_VALID_TASKS = N_TEST_TASKS = 500
        N_VALID_TASKS = N_TEST_TASKS = 30


    elif 'mhc' in dataset:
        cv_split_id = int(dataset[-1])
        dataset = MHCMetaDataset(cv_split_id=cv_split_id, random_state=np.random.RandomState(seed + 1))
        if n_train_tasks is None:
            n_train_tasks = 5
        if n_samples is None:
            n_train_samples = -1
            n_samples_context = 20

        data_train = dataset.generate_meta_train_data(n_tasks=n_train_tasks, n_samples=n_train_samples)
        data_test_valid = dataset.generate_meta_test_data(n_samples_context=n_samples_context,
                                                          n_samples_test=-1)

        # the mhc data doesn't have enough datasets to allow for a proper valid / test split
        return data_train, data_test_valid, data_test_valid

    elif 'berkeley' in dataset:
        if len(dataset.split('_')) == 2:
            n_train_tasks = int(dataset.split('_')[-1])

        dataset = BerkeleySensorMetaDataset(random_state=np.random.RandomState(seed + 1))

        assert n_samples is None
        n_train_samples = 2*144
        n_samples_context = 144 # corresponds to first day of measurements
        data_train = dataset.generate_meta_train_data(n_tasks=n_train_tasks, n_samples=n_train_samples)
        data_test_valid = dataset.generate_meta_test_data(n_samples_context=n_samples_context,
                                                          n_samples_test=-1)
        return data_train, data_test_valid, data_test_valid


    elif dataset == 'swissfel':
        dataset = SwissfelMetaDataset(random_state=np.random.RandomState(seed + 1))
        if n_train_tasks is None:
            n_train_tasks = 5

        if n_samples is None:
            n_train_samples = n_context_samples = 200
        else:
            n_train_samples = n_context_samples = n_samples

        N_TEST_SAMPLES = 200

        data_train = dataset.generate_meta_train_data(n_tasks=n_train_tasks, n_samples=n_train_samples)

        data_test_valid = dataset.generate_meta_test_data(n_samples_context=n_context_samples,
                                                          n_samples_test=N_TEST_SAMPLES)

        # swissfel data doesn't have enough datasets to allow for a proper valid / test split
        return data_train, data_test_valid, data_test_valid

    else:
        raise NotImplementedError('Does not recognize dataset flag')

    data_train = dataset.generate_meta_train_data(n_tasks=n_train_tasks, n_samples=n_train_samples)

    data_test_valid = dataset.generate_meta_test_data(n_tasks=N_TEST_TASKS + N_VALID_TASKS,
                                                      n_samples_context=n_context_samples,
                                                      n_samples_test=N_TEST_SAMPLES)
    data_valid = data_test_valid[N_VALID_TASKS:]
    data_test = data_test_valid[:N_VALID_TASKS]

    return data_train, data_valid, data_test


