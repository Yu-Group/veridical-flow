VERSION ?= $(shell git rev-parse --short HEAD)
CONDA_ENV_NAME ?= vflow
HATCH_ENV_NAME ?= test

build_conda_env:
	conda create -n $(CONDA_ENV_NAME) -y python==3.10 pip
	conda run -n $(CONDA_ENV_NAME) --no-capture-output ./pip_install.sh

build_ipykernel: build_conda_env
	conda run -n $(CONDA_ENV_NAME) python -m ipykernel install --user --name $(CONDA_ENV_NAME) --display-name "Python [conda:$(CONDA_ENV_NAME)]"

build_hatch_env_%:
	hatch -v env create $*

test_%: build_hatch_env_test
	hatch -v run test:pytest $(PYTEST_ARGS) tests/test_$*.py

run_tests: build_hatch_env_test
	hatch -v run test:coverage run --source=vflow,tests -m pytest $(PYTEST_ARGS) tests
	hatch -v run test:coverage report -m

fix_styles: build_hatch_env_style
	hatch -v run style:fmt
