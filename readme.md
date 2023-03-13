<header>
<p align="center">
	<img src="https://yu-group.github.io/veridical-flow/logo_vflow_straight.png" width="70%" alt="vflow logo"> 
</p>

  
<p align="center"> A library for making stability analysis simple. Easily evaluate the effect of judgment calls to your data-science pipeline (e.g. choice of imputation strategy)!
</p>
<p align="center">
  <img src="https://img.shields.io/badge/license-mit-blue.svg" alt="mit license">
  <img src="https://img.shields.io/badge/python-3.6+-blue" alt="python3.6+">
  <a href="https://github.com/Yu-Group/veridical-flow/actions"><img src="https://github.com/Yu-Group/veridical-flow/workflows/tests/badge.svg" alt="tests"></a>
  <a href="https://app.codecov.io/gh/Yu-Group/veridical-flow/commits?page=1"><img src="https://codecov.io/gh/Yu-Group/veridical-flow/branch/master/graph/badge.svg?token=YUAKU54XS4" alt="tests"></a>
  <a href="https://joss.theoj.org/papers/10.21105/joss.03895"><img src="https://joss.theoj.org/papers/10.21105/joss.03895/status.svg" alt="joss"></a>
  <img src="https://img.shields.io/pypi/v/vflow?color=orange" alt="downloads">
</p> 
</header>



# Why use `vflow`?

Using `vflow`s simple wrappers facilitates many best practices for data science,
as laid out in the predictability, computability, and stability (PCS) framework
for [veridical data science](https://www.pnas.org/content/117/8/3920). The goal
of `vflow` is to easily enable data science pipelines that follow PCS by
providing intuitive low-code syntax, efficient and flexible computational
backends via [`Ray`](https://docs.ray.io/en/latest/ray-core/walkthrough.html),
and well-documented, reproducible experimentation via
[`MLflow`](https://mlflow.org/docs/latest/index.html).

| Computation | Reproducibility | Prediction | Stability |
| ----------- | --------------- | ---------- | --------- |
| Automatic parallelization and caching throughout the pipeline | Automatic experiment tracking and saving | Filter the pipeline by training and validation performance | Replace a single function (e.g. preprocessing) with a set of functions and easily assess the stability of downstream results |

Here we show a simple example of an entire data-science pipeline with several
perturbations (e.g. different data subsamples, models, and metrics) written
simply using `vflow`.

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

Once we've written this pipeline, we can easily measure the stability of metrics (e.g. "Accuracy") to our choice of subsampling or model.

# Documentation

See the [docs](https://yu-group.github.io/veridical-flow/) for reference on the API

> **Notebook examples** (Note that some of these require more dependencies than just those required for vflow - to install all, use the `notebooks` dependencies in the `setup.py` file)
>
> [Synthetic classification](https://yu-group.github.io/veridical-flow/notebooks/00_synthetic_classification.html)
>
> [Enhancer genomics](https://yu-group.github.io/veridical-flow/notebooks/01_enhancer.html)
>
> [fMRI voxel prediction](https://yu-group.github.io/veridical-flow/notebooks/02_fmri.html)
> 
> [Fashion mnist classification](https://yu-group.github.io/veridical-flow/notebooks/03_computer_vision_dnn.html)
>
> [Feature importance stability](https://yu-group.github.io/veridical-flow/notebooks/04_feat_importance_stability.html)
> 
> [Clinical decision rule vetting](https://github.com/Yu-Group/rule-vetting)

## Installation

Install with `pip install vflow` (see [here](https://github.com/Yu-Group/veridical-flow/blob/master/docs/troubleshooting.md) for help). For dev version (unstable), clone the repo and run `python setup.py develop` from the repo directory.

# References

- interface: easily build on [scikit-learn](https://scikit-learn.org/stable/index.html) and [dvc](https://dvc.org/) (data version control)
- computation: integration with [ray](https://www.ray.io/) and caching with [joblib](https://joblib.readthedocs.io/en/latest/)
- tracking: [mlflow](https://mlflow.org/)
- pull requests very welcome! (see [contributing.md](https://github.com/Yu-Group/veridical-flow/blob/master/docs/contributing.md))

```r
@software{duncan2020vflow,
   author = {Duncan, James and Kapoor, Rush and Agarwal, Abhineet and Singh, Chandan and Yu, Bin},
   doi = {10.21105/joss.03895},
   month = {1},
   title = {{VeridicalFlow: a Python package for building trustworthy data science pipelines with PCS}},
   url = {https://doi.org/10.21105/joss.03895},
   year = {2022}
}
```
