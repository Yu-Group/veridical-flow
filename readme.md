<h1 align="center"> PCS Pipeline </h1>
<p align="center"> A framework for doing stability analysis with PCS. (Under development, not ready for use)
</p>

<p align="center">
  <img src="https://img.shields.io/badge/license-mit-blue.svg">
  <img src="https://img.shields.io/badge/python-3.6--3.8-blue">
  <a href="https://github.com/Yu-group/pcs-pipeline/actions"><img src="https://github.com/Yu-group/pcs-pipeline/workflows/tests/badge.svg"></a>
  <img src="https://img.shields.io/github/checks-status/Yu-group/pcs-pipeline/master">
</p>  
<p align="center">
    <a href="https://yu-group.github.io/pcs-pipeline/">Docs</a>
</p>  


# Sample usage

Install with `pip install pcsp` (see [here](https://github.com/Yu-Group/pcs-pipeline/blob/master/docs/troubleshooting.md) for help)

```python
import pcsp
from pcsp import PCSPipeline # replaces sklearn.Pipeline
```

# Documentation

Builds heavily on the [sklearn-pipeline](https://scikit-learn.org/stable/modules/compose.html), but extends it to facilitate stability analysis.

## Pipeline

The [`Pipeline`](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html#sklearn.pipeline.Pipeline) is built using a list of `(key, value)` pairs, where the `key` is a string containing the name you want to give this step and `value` is an estimator object:

```python
from pcsp import PCSPipeline # replaces sklearn.Pipeline
pipe = PCSPipeline()
```

The estimators of a pipeline are stored as a list in the steps attribute, but can be accessed by index or name by indexing (with [idx]) the Pipeline:

```python
>>> pipe.steps[0]
('reduce_dim', PCA())
>>> pipe[0]
PCA()
```

> **Examples**
>
>  [Digit classification pipeline example](notebooks/digits_classification.ipynb)


# References

- built on scikit-learn
- compatible with [dvc](https://dvc.org/) - data version control
- uses [joblib](https://joblib.readthedocs.io/en/latest/) - run functions as pipeline jobs
- pull requests <a href="https://github.com/Yu-Group/pcs-pipeline/blob/master/docs/contributing.md">very welcome</a>!