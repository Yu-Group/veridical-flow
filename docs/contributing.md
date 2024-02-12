Contributing to VeridicalFlow
=======

Contributions are very welcome!

## Getting Started

If you are looking to contribute to the `vflow` codebase, the best place to
start is the [GitHub "issues"
tab](https://github.com/Yu-Group/veridical-flow/issues). This is also a great
place for filing bug reports and suggesting improvement to the code and
documentation.

## Filing Issues

If you notice a bug in the code or documentation, or have suggestions for how we
can improve either, feel free to create an issue on the [GitHub "issues" tab](https://github.com/Yu-Group/veridical-flow/issues) using [GitHub's "issue" form](https://github.com/Yu-Group/veridical-flow/issues/new). Please be as
descriptive as possible we so can best address your issue.

## Making contributions

We use [Hatch](https://hatch.pypa.io/latest/) for development workflow
management. To get started, install Hatch and [fork the
repository](https://github.com/Yu-Group/veridical-flow/fork). Before submitting
your changes for review by [pull
request](https://github.com/Yu-Group/veridical-flow/compare), please add
full-coverage tests and make sure to check that they do not break any existing
tests. In addition, be sure to conform your code styles to our conventions and
document any updates to the public API.

- [Tests](https://github.com/Yu-Group/veridical-flow/tree/master/tests): Tests are written with [pytest](https://docs.pytest.org/en/stable/)
  and use [Coverage.py](https://coverage.readthedocs.io/en/latest/). Run the
  command `hatch run cov` from the repo root directory.
- Code styles: We use [isort](https://pycqa.github.io/isort/) and
  [Black](https://black.readthedocs.io/en/stable/) to automate imports
  organizing and code formatting, along with
  [ruff](https://docs.astral.sh/ruff/) to passively check for issues. You can
  use `hatch run style:fmt` from the repo root directory to format and check
  your changes.
- [Docs](https://github.com/Yu-Group/veridical-flow/tree/master/docs): To build
  the documentation, run `hatch run docs:build` from the repo root directory.
  Note that we follow the [numpy
  docstrings](https://numpydoc.readthedocs.io/en/latest/format.html) convention.

See the
[Makefile](https://github.com/Yu-Group/veridical-flow/blob/master/Makefile) for
other helpful development workflow commends.

Once your changes are ready to be submitted, make sure to push your change to
GitHub before creating a pull request. See
[here](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork)
for instruction on how to create a pull request from a fork. We will review your
changes, and you may be asked to make additional changes before it is finally
ready to merge. Once it is ready, we will merge your pull request, and you have
will successfully contributed to VeridicalFlow 🎉!
