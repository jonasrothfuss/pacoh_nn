## PACOH-NN: Meta-Learning Bayesian Neural Network Priors based on PAC-Bayesian Theory
This repository provides source code for the *PACOH-NN* method, introduced in the paper [*PACOH: Bayes-Optimal Meta-Learning with PAC-Guarantees*](https://arxiv.org/abs/2002.05551).

Authors of the repository: [Martin Josifoski](https://people.epfl.ch/martin.josifoski/?lang=en) and [Jonas Rothfuss](https://las.inf.ethz.ch/people/jonas-rothfuss) 
## Installation
To install the minimal dependencies needed to use the meta-learning algorithms, run in the main directory of this repository
```bash
pip install .
``` 

Alternatively, you can install the required packages using the following command:
```bash
pip install -r requirements.txt
``` 

## Usage
The following code snippet demonstrates the core functionality of the meta-learners provided in this repository. 
In addition, **run_cauchy.py** provides an concise example how to run PACOH-NN with the Cauchy mete-learning environment fro the paper.
Finally, **demo_pacoh_nn.ipynb** contains a detailed demo code with plots and discussions. 
This is probably the best point to dive into the code.

```python
""" A) generate meta-training and meta-testing data """
from pacoh_nn.datasets import provide_data
meta_train_data, _, meta_test_data = provide_data(dataset='sin_20')

""" B) Meta-Learning with PACOH-NN """
from pacoh_nn.pacoh_nn_regression import PACOH_NN_Regression
pacoh_model = PACOH_NN_Regression(meta_train_data, random_seed=22, num_iter_meta_train=20000,
                                  learn_likelihood=True, likelihood_std=0.1)
pacoh_model.meta_fit(meta_test_data[:10], eval_period=5000, log_period=1000)


"""  C) Meta-Testing with PACOH-NN """
x_context, y_context, x_test, y_test = meta_test_data[0]

# target training in (x_ontext, y_context) & predictions for x_test
y_preds, pred_dist = pacoh_model.meta_predict(x_context, y_context, x_test)

# compute evaluation metrics on one target task
eval_metric_dict = pacoh_model.meta_eval(x_context, y_context, x_test, y_test, y_test)

# compute evaluation metrics for multiple tasks / test datasets
eval_metrics_mean_dict, eval_metrics_std_dict = pacoh_model.meta_eval_datasets(meta_test_data)
```
## Loading the meta-learning datasets / environments
The meta-learning regression environments that were used in the paper can be loaded using
 the ```provide_data``` function:
```python
from pacoh_nn.datasets import provide_data
meta_train_data, meta_valid_data, meta_test_data = provide_data(dataset=DATASET_NAME)
```
Once you call provide_data, the necessary datasets are automatically downloaded. The following table maps the environment name in the paper to the the DATASET_NAME which needs to be used in the code:

| Name in the paper        | DATASET_NAME   |
| ------------- |:-------------:|
| Sinosoids   | sin_20 | 
| Cauchy    | cauchy_20      |
| Swissfel | swissfel  |
| Physionet-GCS | physionet_0  | 
| Physionet-HCT | physionet_2  | 
| Berkeley-Sensor | berkeley  | 

## Citing
If you use the PACOH-NN implementation or the meta-learning environments in your research, please cite it as follows:

```
@inproceedings{rothfuss2021pacoh,
  title={{PACOH: Bayes-Optimal Meta-Learning with PAC-Guarantees}},
  author={Rothfuss, Jonas and Fortuin, Vincent and Josifoski, Martin and Krause, Andreas},
  booktitle={International Conference on Machine Learning},
  year={2021}
}
```
