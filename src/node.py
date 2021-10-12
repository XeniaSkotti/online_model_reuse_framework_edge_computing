import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM

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

def get_node_data(data, experiment):
    exp = data.loc[data.experiment==experiment][["humidity", "temperature"]]
    a = exp.loc[data.pi=="pi2"]
    b = exp.loc[data.pi=="pi3"]
    c = exp.loc[data.pi=="pi4"]
    d = exp.loc[data.pi =="pi5"]
    return remove_outliers([a,b,c,d])

def remove_outliers(node_data):
    for i in range(4):
        model = OneClassSVM(nu=0.1)
        pred = model.fit_predict(node_data[i])
        inliers = np.where(pred == 1)
        node_data[i] = node_data[i].iloc[inliers]
        
    return node_data