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

# the following commands can only be used when `make` was executed successfully
# TODO issue-135: print a hint on the console to execute `make`
jupyter:
	.venv/bin/jupyter notebook --no-browser

mkdocs:
	.venv/bin/mkdocs serve

format:
	.venv/bin/black .

type-check:
	.venv/bin/mypy src/

lint: markdownlint
	.venv/bin/ruff check --fix
	markdownlint --fix '**/*.md'

pytest:
	.venv/bin/pytest

assignment-1:
	.venv/bin/pytest tests/htwgnlp/test_preprocessing.py

assignment-2:
	.venv/bin/pytest tests/htwgnlp/test_features.py
	.venv/bin/pytest tests/htwgnlp/test_logistic_regression.py

assignment-3:
	.venv/bin/pytest tests/htwgnlp/test_naive_bayes.py

assignment-4:
	.venv/bin/pytest tests/htwgnlp/test_embeddings.py
