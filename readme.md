<p align="center">
	<img src="https://yu-group.github.io/veridical-flow/logo_vflow_straight.jpg" width="70%"> 
</p>


<p align="center"> A library for making stability analysis simple, following the veridical data-science framework.
</p>
<p align="center">
  <img src="https://img.shields.io/badge/license-mit-blue.svg">
  <img src="https://img.shields.io/badge/python-3.6+-blue">
  <a href="https://github.com/Yu-group/pcs-pipeline/actions"><img src="https://github.com/Yu-group/pcs-pipeline/workflows/tests/badge.svg"></a>
  <img src="https://img.shields.io/github/checks-status/Yu-group/pcs-pipeline/master">
  <img src="https://img.shields.io/pypi/v/vflow?color=orange">
</p> 
# Why use `vflow`?

Using `vflow`'s simple wrappers easily enables many best practices for data science, and makes writing pipelines easy.

| Stability                                                    | Computation                                                  | Reproducibility                          |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ---------------------------------------- |
| Replace a single function (e.g. preprocessing) with a set of functions and easily assess the stability of downstream results | Automatic parallelization and caching throughout the pipeline | Automatic experiment tracking and saving |

Here we show a simple example of an entire data-science pipeline with several perturbations (e.g. different data subsamples, models, and metrics) written simply using `vflow`.

```python
import sklearn
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from vflow import init_args, Vset

# initialize data
X, y = sklearn.datasets.make_classification()
X_train, X_test, y_train, y_test = init_args(
    sklearn.model_selection.train_test_split(X, y),
    names=['X_train', 'X_test', 'y_train', 'y_test']  # optionally name the args
)

# subsample data
subsampling_funcs = [
    sklearn.utils.resample for _ in range(3)
]
subsampling_set = Vset(name='subsampling',
                       modules=subsampling_funcs,
                       output_matching=True)
X_trains, y_trains = subsampling_set(X_train, y_train)

# fit models
models = [
    sklearn.linear_model.LogisticRegression(),
    sklearn.tree.DecisionTreeClassifier()
]
modeling_set = Vset(name='modeling',
                    modules=models,
                    module_keys=["LR", "DT"])
modeling_set.fit(X_trains, y_trains)
preds_test = modeling_set.predict(X_test)

# get metrics
binary_metrics_set = Vset(name='binary_metrics',
                          modules=[accuracy_score, balanced_accuracy_score],
                          module_keys=["Acc", "Bal_Acc"])
binary_metrics = binary_metrics_set.evaluate(preds_test, y_test)
```

Once we've written this pipeline, get very easily see how stable certain metrics (e.g. "Acc") are to our choice of subsampling or model.

# Documentation

See the [docs](https://yu-group.github.io/veridical-flow/) for reference on the API

> **Examples**
>
> [Synthetic classification example](https://github.com/Yu-Group/veridical-flow/blob/master/notebooks/00_synthetic_classification.ipynb)
>
> [Enhancer example](https://github.com/Yu-Group/veridical-flow/blob/master/notebooks/01_enhancer.ipynb)
>
> [fMRI example](https://github.com/Yu-Group/veridical-flow/blob/master/notebooks/02_fmri.ipynb)

## Installation

Install with `pip install vflow` (see [here](https://github.com/Yu-Group/pcs-pipeline/blob/master/docs/troubleshooting.md) for help). For dev version (unstable), clone the repo and run `python setup.py develop` from the repo directory.

# References

- interface: easily build on [scikit-learn](https://scikit-learn.org/stable/index.html) and [dvc](https://dvc.org/) (data version control)
- computation: integration with [ray](https://www.ray.io/) and caching with [joblib](https://joblib.readthedocs.io/en/latest/)
- tracking: [mlflow](https://mlflow.org/)
- pull requests very welcome! (see [contributing.md](https://github.com/Yu-Group/pcs-pipeline/blob/master/docs/contributing.md))