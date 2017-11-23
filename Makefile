develop:
	python setup.py develop

bdist:
	python setup.py bdist_wheel

install: bdist
	pip install dist/netensorflow-*.whl --upgrade

uninstall:
	pip uninstall -y netensorflow
