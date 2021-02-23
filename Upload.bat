rem run 'where anaconda' and save the resulting path into @ana
@echo off
for /f %%i in ('where conda') do set @ana=%%i

rem set p_dir to be a string activating the conda environment
@echo off
set _path=%@ana%
for %%a in ("%_path%") do set "p_dir=%%~dpa"

SET p_dir=%p_dir%activate

rem activate the conda environment
call %p_dir%

if %errorlevel% neq 0 call C:\Users\%USERNAME%\AppData\Local\Continuum\anaconda3\Scripts\activate

rem install virtualenv
call pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org virtualenv

rem create a virtual environment in the code folder
rem call virtualenv Code\venv
call python -m venv Code\venv

rem activate this new virtual environment
call Code\venv\Scripts\activate

rem if there is an error at this point, don't install anything
if %errorlevel% neq 0 exit

rem install the accompanying requirements file
rem call pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r Code\requirements.txt

rem install jupyter into the virtual environment
rem call pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org jupyter

rem install ipython into the virtual environment
rem call pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org ipython

rem add the virtual environment as a kernel in the local jupyter notebook
rem call ipython kernel install --user --name=venv

rem upload to pypi
call pip install --upgrade setuptools wheel
call python setup.py sdist bdist_wheel
call pip install --upgrade twine
rem call pip install pyOpenSSL ndg-httpsclient pyasn1
call python -m twine upload dist/*

rem leave the window open
cmd /k
