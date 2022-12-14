docs: requirements
	.venv/bin/mkdocs serve

requirements: .venv
	.venv/bin/pip install --upgrade .
	rm -rf ./build/
	rm -rf ./htwg_practical_nlp.egg-info/

.venv:
	python3.10 -m venv --upgrade-deps .venv