Contributing to veridical-flow
=======

Contributions are very welcome!

## Getting Started

If you are looking to contribute to the *veridical-flow* codebase, the best place to start is the [GitHub "issues" tab](https://github.com/Yu-Group/veridical-flow/issues). This is also a great place for filing bug reports and suggesting improvement to the code and documentation.

## Filing Issues

If you notice a bug in the code or documentation, or have suggestions for how we can improve either, feel free to create an issue on the [GitHub "issues" tab](https://github.com/Yu-Group/veridical-flow/issues) using [GitHub's "issue" form](https://github.com/Yu-Group/veridical-flow/issues/new). Please be as descriptive as possible we so can best address your issue.

## Contributing to the Codebase

After forking the [repository](https://github.com/Yu-Group/veridical-flow), be sure to create a development environment that is separate from your existing Python environment so that you can make and test changes without compromising your own work environment. To develop locally, navigate to the repo root directory and run `python setup.py develop --user`.

Before submitting your changes for review, please make sure to check that your changes do not break any tests. 
- [Tests](tests) are written with [pytest](https://docs.pytest.org/en/stable/), and can be run by running `pytest` in the repo root directory. 
- [Docs](https://csinva.io/imodels/docs/) are built using [pdoc](https://pdoc3.github.io/pdoc/), and can be built by navigating to the `docs` directory and running `./build_docs.sh`.

Once your changes are ready to be submitted, make sure to push your change to GitHub before creating a pull request. See [here](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork) for instruction on how to create a pull request from a fork. We will review your changes, and you may be asked to make additional changes before it is finally ready to merge. Once it is ready, we will merge your pull request, and you have will successfully contributed to the codebase!






