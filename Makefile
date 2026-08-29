PYTHON ?= python3
PYTHONPATH := code/src

.PHONY: test toy list-data validate-config pull-cifar10 pull-cifar100 pull-cifar100-hf pull-pilot

test:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m unittest discover -s code/tests -v

toy:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) code/scripts/run_toy.py

list-data:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) code/scripts/pull_data.py --list --manifest pull/manifest.yaml

validate-config:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) code/scripts/validate_config.py code/configs/pilot.yaml

pull-cifar10:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) code/scripts/pull_data.py --dataset cifar10 --root pull/raw

pull-cifar100:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) code/scripts/pull_data.py --dataset cifar100 --root pull/raw

pull-cifar100-hf:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) code/scripts/pull_data.py --dataset cifar100 --backend huggingface --root pull/raw

pull-pilot: pull-cifar100-hf
