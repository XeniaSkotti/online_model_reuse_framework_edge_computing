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

def get_similar_pairs_ocsvm(node_data, models, inliers, threshold, unidirectional):
    similar_pairs = []
    similar_nodes = []
    combos = comb(range(4),2)
    for c in combos:
        x, y = c[0], c[1]

        model_x, model_y = models[x], models[y]
        predicted_x_inliers = np.where(model_y.predict(node_data[x][["humidity", "temperature"]]) == 1)[0]
        predicted_y_inliers = np.where(model_x.predict(node_data[y][["humidity", "temperature"]]) == 1)[0]

        x_y_overlap = np.intersect1d(inliers[x], predicted_x_inliers).shape[0]/inliers[x].shape[0]
        y_x_overlap = np.intersect1d(inliers[y], predicted_y_inliers).shape[0]/inliers[y].shape[0]
        
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
    
    return similar_pairs, similar_nodes