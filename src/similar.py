from maximum_mean_discrepancy import get_tensor_sample, get_tensor_samples, MMD
from maximum_mean_discrepancy import avg_similarity_disimilarity_MMD as ASDMMD
from itertools import combinations as comb
import numpy as np
import pandas as pd

def get_mmd_args(experiment, standardised):
    if experiment ==1:
        if standardised:
            mmd_args = ("rbf", 0.5)
        else:
            mmd_args = ("rbf", 10)
    elif experiment == 2:
        if standardised:
            mmd_args = ("rbf", 1)
        else:
            mmd_args = ("rbf", 100)
    elif experiment == 3:
        if standardised:
            mmd_args = ("rbf", 1)
        else:
            mmd_args = ("rbf", 5)
    return mmd_args

def get_similar_other_nodes_sets(experiment, std=False):
    if experiment == 1:
        if std:
            similar_nodes = ["pi2","pi3","pi4"]
            other_nodes = ["pi5"]
        else:
            similar_nodes = ["pi2", "pi4"]
            other_nodes = ["pi3", "pi5"]
    elif experiment == 2:
        if std:
            similar_nodes = ["pi2", "pi3", "pi5"]
            other_nodes = ["pi4"]
        else:
            similar_nodes = ["pi3", "pi5"]
            other_nodes = ["pi2", "pi4"]
    elif experiment == 3:
        similar_nodes = ["pi3", "pi5"]
        other_nodes = ["pi2", "pi4"]
    return similar_nodes, other_nodes

def is_similar_pair(x,y, asdmmd, kernel, kernel_bandwidth):
    mmd = MMD(x,y, kernel, kernel_bandwidth)
    if mmd < asdmmd + asdmmd * 0.05:
        return True
    else:
        return False

def find_similar_pairs_mmd(node_data, asmmd, kernel, kernel_bandwidth):
    
    """Finds the pairs of nodes which are similar using the ASMMD
    
    Args:
        node_data: list of dataframes, one for each node
        asdmmd: the average similarity MMD
        kernel: the kernel type to be used for the MMD calculation.
        kernel_bandwidth: scalar value to be used by the kernel in the MMD calculation. 
    """
    
    combos = comb(range(4),2)
    similar_pairs = []
    similar_nodes = []
    pairs_mmd =[]
    threshold = asmmd + asmmd * 0.05
    for c in combos:
        node_x = "pi"+str(c[0]+2)
        node_y = "pi"+str(c[1]+2)
        
        x = node_data[c[0]][["humidity", "temperature"]].values.astype(np.float32)
        y = node_data[c[1]][["humidity", "temperature"]].values.astype(np.float32)

        sample_size = min(x.shape[0], y.shape[0])
        tx, ty = get_tensor_sample(x, sample_size), get_tensor_sample(y, sample_size)

        mmd = MMD(tx, ty, kernel, kernel_bandwidth)
        if mmd < threshold:
            if node_x not in similar_nodes:
                similar_nodes.append(node_x)
            if node_y not in similar_nodes:
                similar_nodes.append(node_y)
            pairs_mmd.append(mmd)
                
            similar_pairs.append((node_x, node_y))
    return similar_pairs, similar_nodes, pairs_mmd

def calculate_ocsvm_scores(node_data, similar_pairs, models):
    pair_thresholds = []
    for x, y in similar_pairs:

        model_x, model_y = models[x], models[y]
        
        sample_x = node_data[x][["humidity", "temperature"]]
        sample_y = node_data[y][["humidity", "temperature"]]
        
        predicted_x_inliers = np.where(model_y.predict(sample_x) == 1)[0]
        predicted_y_inliers = np.where(model_x.predict(sample_y) == 1)[0]
        
        x_y_overlap = len(predicted_y_inliers)/len(sample_y)
        y_x_overlap = len(predicted_x_inliers)/len(sample_x)

        pair_thresholds.append((x_y_overlap, y_x_overlap))
    
    return pair_thresholds

def get_similar_pairs_nodes(experiment, data, method, standardised):
    node_data = data[experiment]["sampled_data"]
    models = data[experiment]["models"]
    
    kernel, kernel_bandwidth =  get_mmd_args(experiment, standardised)  
    tensor_samples = get_tensor_samples(node_data, sample_size=node_data[0].shape[0])
    similar_nodes, other_nodes = get_similar_other_nodes_sets(experiment, standardised)
    asmmd = ASDMMD(tensor_samples, similar_nodes, other_nodes, kernel, kernel_bandwidth, return_tables = False)
    
    similar_pairs, similar_nodes, mmd_scores = find_similar_pairs_mmd(node_data, asmmd, kernel, kernel_bandwidth)
    ocsvm_scores = calculate_ocsvm_scores(node_data, similar_pairs, models)

#     if isinstance(similar_pairs, dict):
#         thresholds = {k:i for k, i in thresholds.items() if similar_pairs[k] != []}
#         similar_pairs = {k:i for k, i in similar_pairs.items() if i != []}
#         similar_nodes = {k:i for k, i in similar_nodes.items() if i != []}
    
    return similar_pairs, similar_nodes, mmd_scores, ocsvm_scores