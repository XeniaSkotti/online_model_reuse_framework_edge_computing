from maximum_mean_discrepancy import get_tensor_sample, get_tensor_samples, MMD
from maximum_mean_discrepancy import avg_similarity_disimilarity_MMD as ASDMMD
from node import get_similar_other_nodes_sets, remove_outliers, get_node_data
from itertools import combinations as comb
import numpy as np
import pandas as pd

def is_similar_pair(x,y, asdmmd, kernel, kernel_bandwidth):
    mmd = MMD(x,y, kernel, kernel_bandwidth)
    if mmd < asdmmd + asdmmd * 0.05:
        return True
    else:
        return False

def find_similar_pairs_mmd(node_data, asmmd, kernel, kernel_bandwidth):
    
    """Finds the pairs of nodes which are similar using the ASMMD
    
    Args:
        node_data: list of dataframes, one for each no
        asdmmd: the average similarity MMD
        kernel: the kernel type to be used for the MMD calculation.
        kernel_bandwidth: scalar value to be used by the kernel in the MMD calculation. 
    """
    
    combos = comb(range(4),2)
    similar_pairs = []
    similar_nodes = []
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
                
            similar_pairs.append((node_x, node_y))
    return similar_pairs, similar_nodes, threshold

def find_similar_pairs_ocsvm(node_data, models, inliers, threshold):
    similar_pairs = []
    similar_nodes = []
    pair_thresholds = []
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
                
            similar_pairs.append((node_x, node_y))
            pair_thresholds.append((x_y_overlap, y_x_overlap))
    
    return similar_pairs, similar_nodes, pair_thresholds

def get_mmd_similar_pairs(data, experiment, kernel, kernel_bandwidth):
    node_data = data[experiment]["sampled_data"]
    
    tensor_samples = get_tensor_samples(node_data, sample_size=node_data[0].shape[0])
    similar_nodes, other_nodes = get_similar_other_nodes_sets(experiment)
    asmmd = ASDMMD(tensor_samples, similar_nodes, other_nodes, kernel, kernel_bandwidth, return_tables = False)
#     print(f"The average MMD between similar sets is {asmmd}")
    
    similar_pairs, similar_nodes, threshold = find_similar_pairs_mmd(node_data, asmmd, kernel, kernel_bandwidth)
#     print(f"The following pairs of nodes were deemed similar {similar_pairs} using the MMD method.\n")
    
    return similar_pairs, similar_nodes, threshold

def get_ocsvm_similar_pairs(data, experiment, threshold):
    node_data = get_node_data(data[experiment]["raw_data"], experiment, filtered = False)
    models = data[experiment]["models"]
    inliers =  data[experiment]["inliers"]
    similar_pairs, similar_nodes, pair_thresholds = find_similar_pairs_ocsvm(node_data, models, inliers, threshold)
#     print(f"The following pairs of nodes were deemed similar {similar_pairs} using the OCSVM method.\n")
    
    return similar_pairs, similar_nodes, pair_thresholds

def get_similar_pairs_nodes(experiment, data, method, similar_pairs_args):
    if method in ["mmd", "ocsvm"]:
        if method == "mmd":
            kernel, kernel_bandwidth = similar_pairs_args
            similar_pairs, similar_nodes, thresholds = get_mmd_similar_pairs(data, experiment, kernel, kernel_bandwidth)
        elif method == "ocsvm":
            threshold = similar_pairs_args
            similar_pairs, similar_nodes, thresholds = get_ocsvm_similar_pairs(data, experiment, threshold)  
    elif method in ["both", "verify", "trio"]:
        threshold = similar_pairs_args[1]
        ocsvm_similar_pairs, ocsvm_similar_nodes, ocsvm_thresholds = get_ocsvm_similar_pairs(data, experiment, threshold)  

        kernel, kernel_bandwidth = similar_pairs_args[0]
        mmd_similar_pairs, mmd_similar_nodes, mmd_threshold = get_mmd_similar_pairs(data, experiment, kernel, kernel_bandwidth)
        
        if method in ["both", "trio"]:
            similar_pairs = {"MMD" : mmd_similar_pairs, "OCSVM" : ocsvm_similar_pairs}
            similar_nodes = {"MMD" : mmd_similar_nodes, "OCSVM" : ocsvm_similar_nodes}
            thresholds = {"MMD" : mmd_threshold, "OCSVM" : ocsvm_thresholds}
        
        if method in  ["verify", "trio"]:
            str_mmd_similar_pairs = str([pair[::-1] for pair in mmd_similar_pairs]) + str(mmd_similar_pairs)
            verify_similar_pairs = []
            verify_thresholds = []
            for pair_index in range(len(ocsvm_similar_pairs)): 
                pair = ocsvm_similar_pairs[pair_index]
                if str(pair) in str_mmd_similar_pairs:
                    verify_similar_pairs.append(pair)
                    verify_thresholds.append(ocsvm_thresholds[pair_index]) 
            verify_similar_nodes = [node for node in ocsvm_similar_nodes if node in str(verify_similar_pairs)]

            if method == "trio":
                similar_pairs["MMD OCSVM Verify"] = verify_similar_pairs
                similar_nodes["MMD OCSVM Verify"] = verify_similar_nodes
                thresholds["MMD OCSVM Verify"] = verify_thresholds
            else:
                similar_pairs = verify_similar_pairs
                similar_nodes = verify_similar_nodes
                thresholds = verify_thresholds
    
    if isinstance(similar_pairs, dict):
        similar_pairs = {k:i for k, i in similar_pairs.items() if i != []}
        similar_nodes = {k:i for k, i in similar_nodes.items() if i != []}
        thresholds = {k:i for k, i in thresholds.items() if similar_pairs[k] != []}
    
    return similar_pairs, similar_nodes, thresholds