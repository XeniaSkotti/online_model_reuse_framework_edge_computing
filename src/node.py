import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM
from itertools import combinations as comb

def get_similar_other_nodes_sets(experiment):
    if experiment == 1:
        similar_nodes = ["pi2", "pi4"]
        other_nodes = ["pi3", "pi5"]
    elif experiment == 2:
        similar_nodes = ["pi3", "pi5"]
        other_nodes = ["pi2", "pi4"]
    elif experiment == 3:
        similar_nodes = ["pi3", "pi5"]
        other_nodes = ["pi2", "pi4"]
    return similar_nodes, other_nodes

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
        inliers = []
    for i in range(4):
        model = OneClassSVM(nu=0.1)
        pred = model.fit_predict(node_data[i][["humidity", "temperature"]])
        node_inliers = np.where(pred == 1)
        node_data[i] = node_data[i].iloc[node_inliers]
        if return_models:
            models.append(model)
            inliers.append(node_inliers[0])
        
    if return_models:
        return node_data, models, inliers
    else:
        return node_data