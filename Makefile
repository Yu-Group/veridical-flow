VERSION ?= $(shell git rev-parse --short HEAD)
CONDA_ENV_NAME ?= vflow
HATCH_ENV_NAME ?= test

build_conda_env:
	conda create -n $(CONDA_ENV_NAME) -y python==3.10 pip
	conda run -n $(CONDA_ENV_NAME) --no-capture-output pip install -r requirements.txt
	conda run -n $(CONDA_ENV_NAME) --no-capture-output pip install . ipykernel

build_ipykernel: build_conda_env
	conda run -n $(CONDA_ENV_NAME) python -m ipykernel install --user --name $(CONDA_ENV_NAME) --display-name "Python [conda:$(CONDA_ENV_NAME)]"

test_%:
	hatch -v run dev $(PYTEST_ARGS) tests/test_$*.py

run_tests:
	hatch -v run cov

fix_styles:
	hatch -v run style:fmt
