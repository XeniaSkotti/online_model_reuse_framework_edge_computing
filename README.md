# master_project

## Virtual Environments

### Creatiion & Activation:

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


