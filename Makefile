project: requirements

requirements: pip
	.venv/bin/python3 -m pip install -e .
	.venv/bin/python3 -m nltk.downloader -d .venv/nltk_data popular

dev-setup: requirements-dev
	.venv/bin/pre-commit install

requirements-dev: pip
	.venv/bin/python3 -m pip install -e .[dev]
	.venv/bin/python3 -m nltk.downloader -d .venv/nltk_data popular

pip: venv
	.venv/bin/pip install --upgrade pip

venv:
	python3 -m venv --upgrade-deps .venv

jupyter: project
	.venv/bin/jupyter notebook --no-browser

docs: project
	.venv/bin/mkdocs serve

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
