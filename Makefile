docs: requirements
	.venv/bin/mkdocs serve

requirements: .venv
	.venv/bin/pip install --upgrade .
	rm -rf ./build/
	rm -rf ./src/htwg_practical_nlp.egg-info/
	.venv/bin/python3 -m nltk.downloader popular

.venv:
	python3.10 -m venv --upgrade-deps .venv