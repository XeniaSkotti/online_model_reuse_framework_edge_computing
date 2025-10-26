# Online Model Reuse Framework in Edge Computing


This repository implements the ideas of the online model reuse framework described in the paper "On the Reusability of Machine Learning Models in Edge Computing: A Statistical Learning Approach" by Xenia Skotti, Kostas Kolomvatsos, and Christos Anagnostopoulos (FTC 2022). The repo contains the data and notebooks showing full development cycle of the framework from it's inception and carrying out it's initial experiments to the paper and presentation showcasing the results.

## Motivation

Edge devices increasingly collect and process large amounts of sensor data. Training machine learning models from scratch on each node is resource‑intensive, and many of these nodes will likely process similar data. Model reuse can reduce training time by leveraging pre‑trained models. However, deciding which models are suitable for reuse on other nodes requires quantifying similarity between datasets and determining the direction of reuse. This repository provides an online framework that uses statistical learning methods to identify good reusability pairs among nodes and choose the best model to reuse

## Key Concepts

- Dataset similarity via MMD: The framework computes the Maximum Mean Discrepancy (MMD) statistic to measure the similarity between datasets
> Pairs of nodes with low MMD are considered candidates for model reuse.

- Direction of reusability: To determine which model should be reused within a pair, the framework measures the overlap of the inlier data spaces using one‑class SVMs
> The model from the node with the higher inlier overlap is selected as the replacement.

- Decision algorithm: An online algorithm examines all pairs of nodes, infers similar pairs based on MMD, and identifies which node's model can be reused for each pair
> A separate decision‑making algorithm maximizes the number of nodes that can reuse models, improving speedup

## Datasets and Models

The framework is evaluated on two datasets:

- GNFUV Sensor Dataset (regression): Contains humidity and temperature readings from four unmanned surface vehicles (USVs). Support Vector Regression (SVR) models are trained to predict the relationship between these variables

- UCI Bank Marketing Dataset (classification): Contains telephone marketing data. The dataset is clustered into nodes using K‑means to simulate multiple edge devices
. Logistic regression models are trained to classify whether a client subscribes to a term deposit.

Experiments consider original and standardised versions of the GNFUV data, and balanced and unbalanced versions of the Bank Marketing data

## Metrics and Results

Two metrics assess the framework’s effectiveness:

- The speedup we benefit from when we avoid training models for some nodes in the network
- the precision of the framework in terms of the recommendations it makes.

Experiments show:
- >0.8 precision when using a tolerance margin (0.05)
- ~30% speed up : 26% on the GNFUV dataset and 29% to 41% for the Bank Marketing dataset,

## Getting Started

Packages listed in requirements.txt (e.g., numpy, pandas, scikit‑learn)

Installation
### Clone the repository
git clone https://github.com/XeniaSkotti/online_model_reuse_framework_edge_computing.git
cd online_model_reuse_framework_edge_computing

### Install dependencies
pip install -r requirements.txt

### Virtual Environments 

#### Creation & Activation:

pip install virtualenv

Go to to the directory you wish to create the virtual environment and do:

	virtualenv <env_name>

To activate the environment do:

	<env_name>\Scripts\activate.bat

#### Regitsering the environment as a Jyputer kernel

if not already installed download ipykernel:

	pip install --user ipykernel

Register the environment using the following:

	python -m ipykernel install --user --name=<env_name>

If you want to delete the virtual environment you should delete it from the list of kernels by:

	jupyter kernelspec uninstall <env_name>

### Jupyter Extensions

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

## Citation

If you use this framework in academic work, please cite the original paper:

Xenia Skotti, Kostas Kolomvatsos, and Christos Anagnostopoulos. “On the Reusability of Machine Learning Models in Edge Computing: A Statistical Learning Approach.” Proceedings of the Future Technologies Conference (FTC) 2022, Lecture Notes in Networks and Systems 2022, DOI: https://doi.org/10.1007/978-3031-18344-7\_5.



