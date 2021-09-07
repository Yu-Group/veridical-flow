<h1 align="center"> Veridical Flow 🌊 </h1>
<p align="center"> A library for making stability analysis simple (following the veridical data science framework).
</p>

<p align="center">
  <img src="https://img.shields.io/badge/license-mit-blue.svg">
  <img src="https://img.shields.io/badge/python-3.6+-blue">
  <a href="https://github.com/Yu-group/pcs-pipeline/actions"><img src="https://github.com/Yu-group/pcs-pipeline/workflows/tests/badge.svg"></a>
  <img src="https://img.shields.io/github/checks-status/Yu-group/pcs-pipeline/master">
  <img src="https://img.shields.io/pypi/v/vflow?color=orange">
</p> 


# Sample usage

Install with `pip install vflow` (see [here](https://github.com/Yu-Group/pcs-pipeline/blob/master/docs/troubleshooting.md) for help). For developer (unstable) version, clone the repo and run `python setup.py develop` from the repo directory.

```python
import vflow
from vflow import PCSPipeline  # replaces sklearn.Pipeline
from vflow import ModuleSet  # drop-in replacement for any function with a set of functions
```

# Documentation

Builds heavily on the [sklearn-pipeline](https://scikit-learn.org/stable/modules/compose.html), but extends it to facilitate stability analysis.

## Pipeline

The [`Pipeline`](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html#sklearn.pipeline.Pipeline) is built using a list of `(key, value)` pairs, where the `key` is a string containing the name you want to give this step and `value` is an estimator object:

```python
from vflow import PCSPipeline  # replaces sklearn.Pipeline

pipe = PCSPipeline()
```

The estimators of a pipeline are stored as a list in the steps attribute, but can be accessed by index or name by indexing (with [idx]) the Pipeline:

```python
>>> pipe.steps[0]
...
>>> pipe[0]
...
```

> **Examples**
>
> [Synthetic classification example](https://github.com/Yu-Group/pcs-pipeline/tree/master/notebooks/synthetic_classification.ipynb)
>
> [Digit classification example](https://github.com/Yu-Group/pcs-pipeline/tree/master/notebooks/digits_classification.ipynb)


# References

- built on [scikit-learn](https://scikit-learn.org/stable/index.html)
- compatible with [dvc](https://dvc.org/) - data version control
- uses [joblib](https://joblib.readthedocs.io/en/latest/) for caching
- pull requests <a href="https://github.com/Yu-Group/pcs-pipeline/blob/master/docs/contributing.md">very welcome</a>!
