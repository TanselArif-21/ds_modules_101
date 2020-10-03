# DS_MODULES_101 Information

## To update on Pypi
1-update version in setup.py
2-start a cmd in the same directory as this file
3-run pip install --user --upgrade setuptools wheel
4-run python setup.py sdist bdist_wheel
5-run pip install --user --upgrade twine
6-run python -m twine upload dist/*