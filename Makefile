VENV = .venv_ds_modules
PYTHON = $(VENV)/bin/python3
PIP = $(VENV)/bin/pip

run: $(VENV)/bin/activate
	#$(PYTHON) dummy.py


$(VENV)/bin/activate: requirements.txt
	#python3 -m venv $(VENV)
	conda create -n $(VENV) python
	#$(PIP) install -r requirements.txt
	conda install --yes --file requirements.txt
	conda activate $(VENV)


clean:
	rm -rf __pycache__
	rm -rf $(VENV)

# run this after activating the environment
jupyter:
	conda install jupyter
	ipython kernel install --name "venv_ds_modules" --user