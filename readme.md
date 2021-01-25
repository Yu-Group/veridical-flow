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
from pcsp import Pipeline
```

# References
- built on scikit-learn
- compatible with [dvc](https://dvc.org/) - data version control
- uses [joblib](https://joblib.readthedocs.io/en/latest/) - run functions as pipeline jobs
- pull requests <a href="https://github.com/Yu-Group/pcs-pipeline/blob/master/docs/contributing.md">very welcome</a>!