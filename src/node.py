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

def get_node_data(data, experiment, filtered = True):
    exp = data.loc[data.experiment==experiment][["humidity", "temperature"]]
    a = exp.loc[data.pi=="pi2"]
    b = exp.loc[data.pi=="pi3"]
    c = exp.loc[data.pi=="pi4"]
    d = exp.loc[data.pi =="pi5"]
    
    node_data = [a,b,c,d]
    if filtered:
        return remove_outliers(node_data)
    else: 
        return node_data

def remove_outliers(node_data, return_models=False):
    if return_models:
        models = []
        inliers = []
    for i in range(4):
        model = OneClassSVM(nu=0.1)
        pred = model.fit_predict(node_data[i])
        node_inliers = np.where(pred == 1)
        node_data[i] = node_data[i].iloc[node_inliers]
        if return_models:
            models.append(model)
            inliers.append(node_inliers)
        
    if return_models:
        return node_data, models, node_inliers
    else:
        return node_data

def get_similar_pairs_unidirectional(models, node_data, threshold, unidirectional):
    similar_pairs = []
    similar_nodes = []
    combos = comb(range(4),2)
    filtered_node_data, models, inliers = remove_outliers(node_data, return_models=True)
    for c in combos:
        x, y = c[0], c[1]

        model_x, model_y = models[x], models[y]
        predicted_x_inliers = np.where(model_y.predict(node_data[x]) == 1)
        predicted_y_inliers = np.where(model_x.predict(node_data[y]) == 1)

        int_x = np.intersect1d(inliers[x], predicted_x_inliers)
        int_y = np.intersect1d(inliers[y], predicted_y_inliers)

        x_y_overlap = int_x.shape[0]/inliers_x.shape[0]
        y_x_overlap = int_y.shape[0]/inliers_y.shape[0]
        
        if max(y_x_overlap,x_y_overlap) > threshold:
            node_x = "pi"+str(x+2)
            node_y = "pi"+str(y+2)
            
            if node_x not in similar_nodes:
                similar_nodes.append(node_x)
            if node_y not in similar_nodes:
                similar_nodes.append(node_y)
                
            if unidirectional:
                if y_x_overlap > threshold:
                    similar_pairs.append((node_x, node_y))
                if x_y_overlap > threshold:
                    similar_pairs.append((node_y, node_x))
            else:
                similar_pairs.append((node_x, node_y))
    
    return similar_pairs, similar_nodes, filtered_node_data