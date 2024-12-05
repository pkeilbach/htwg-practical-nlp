install: requirements

requirements: pip
	.venv/bin/python3 -m pip install -e .
	.venv/bin/python3 -m nltk.downloader -d .venv/nltk_data popular

install-dev: requirements-dev
	.venv/bin/pre-commit install

requirements-dev: pip
	.venv/bin/python3 -m pip install -e .[dev]
	.venv/bin/python3 -m nltk.downloader -d .venv/nltk_data popular

pip: venv
	.venv/bin/pip install --upgrade pip

venv:
	python3 -m venv --upgrade-deps .venv

# General check for any tool in .venv/bin, based on the target name
check-%:
	@if [ ! -f .venv/bin/$* ]; then \
		echo "Tool '$*' not found in .venv/bin. Please run 'make' (or 'make install-dev') to install dependencies"; \
		exit 1; \
	fi

jupyter: check-jupyter
	.venv/bin/jupyter notebook --no-browser

mkdocs: check-mkdocs
	.venv/bin/mkdocs serve

format: check-black
	.venv/bin/black .

type-check: check-mypy
	.venv/bin/mypy src/

lint: check-ruff
	.venv/bin/ruff check --fix
	markdownlint --fix '**/*.md'

pytest: check-pytest
	.venv/bin/pytest

assignment-0: check-pytest
	.venv/bin/pytest tests/htwgnlp/test_python_basics.py

assignment-1: check-pytest
	.venv/bin/pytest tests/htwgnlp/test_preprocessing.py

assignment-2: check-pytest
	.venv/bin/pytest tests/htwgnlp/test_features.py
	.venv/bin/pytest tests/htwgnlp/test_logistic_regression.py

assignment-3: check-pytest
	.venv/bin/pytest tests/htwgnlp/test_naive_bayes.py

assignment-4: check-pytest
	.venv/bin/pytest tests/htwgnlp/test_embeddings.py

check-pre-commit:
	.venv/bin/pre-commit run --all-files
