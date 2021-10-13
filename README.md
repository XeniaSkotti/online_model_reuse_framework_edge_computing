# master_project

## Virtual Environments

### Creation & Activation:

pip install virtualenv

Go to to the directory you wish to create the virtual environment and do:

	virtualenv <env_name>

To activate the environment do:

	<env_name>\Scripts\activate.bat

### Regitsering the environment as a Jyputer kernel

if not already installed download ipykernel:

	pip install --user ipykernel

Register the environment using the following:

	python -m ipykernel install --user --name=<env_name>

If you want to delete the virtual environment you should delete it from the list of kernels by:

	jupyter kernelspec uninstall <env_name>

## Jupyter Extensions

If you want to have added functionality in your Jupyter notebook similar to a Google Collab one install *jupyter_contrib_nbextensions* using: 

	pip install jupyter_contrib_nbextensions

Then execute this command:

	jupyter contrib nbextension install --user

Lastly you need to enable the features you want individually:

	jupyter nbextension enable <nbextension require path>

You can disable features using:

	jupyter nbextension disable <nbextension require path>

The features I've used for this project are the following:

- codefolding/main
- collapsable_headings/main

Full documentation of the extension can be found here:

https://jupyter-contrib-nbextensions.readthedocs.io/en/latest/index.html



