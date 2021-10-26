import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM

def get_node_data(data, experiment, filtered = True, return_models = False):
    exp = data.loc[data.experiment==experiment]
    a = exp.loc[data.pi=="pi2"]
    b = exp.loc[data.pi=="pi3"]
    c = exp.loc[data.pi=="pi4"]
    d = exp.loc[data.pi =="pi5"]
    
    node_data = [a,b,c,d]
    if filtered:
        return remove_outliers(node_data, return_models)
    else: 
        return node_data

def remove_outliers(node_data, return_models=False):
    if return_models:
        models = []
        
    for i in range(4):
        model = OneClassSVM(nu=0.1)
        model.fit(node_data[i][["humidity", "temperature"]])
        if return_models:
            models.append(model)
        
    if return_models:
        return node_data, models
    else:
        return node_data