project: pre_commit nltk

pre_commit: requirements
	.venv/bin/pre-commit install

nltk: requirements
	.venv/bin/python3 -m nltk.downloader -d .venv/nltk_data popular

requirements: pip
	.venv/bin/python3 -m pip install -e .

pip: .venv
	.venv/bin/pip install --upgrade pip

.venv:
	python3.10 -m venv --upgrade-deps .venv

jupyter: project
	.venv/bin/jupyter notebook --no-browser

lecture_notes: project
	.venv/bin/mkdocs serve

pytest:
	.venv/bin/pytest

assignment_1:
	.venv/bin/pytest tests/htwgnlp/test_preprocessing.py

assignment_2:
	.venv/bin/pytest tests/htwgnlp/test_features.py
	.venv/bin/pytest tests/htwgnlp/test_logistic_regression.py

assignment_3:
	.venv/bin/pytest tests/htwgnlp/test_naive_bayes.py

assignment_4:
	.venv/bin/pytest tests/htwgnlp/test_embeddings.py
