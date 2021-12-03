import pandas as pd
import numpy as np 
from results_helper_functions import find_similar_pairs

normalized = {False:"original", True: "standardised"}
kernels = ["rbf", "linear"]
balanced_options = [True, False]

def ocsvm_correct(df, strict):
    if strict:
        max_pairs = df.loc[df["model_r2-d"] == df["model_r2-d"].max()]
    else:
        max_pairs = df.loc[df["model_r2-d"] > df["model_r2-d"].max() - 0.05]
    
    if df["ocsvm_score"].max() in max_pairs["ocsvm_score"].values:
        return 1
    else: 
        return 0

def mmd_correct(df, threshold):
    max_pairs = df.loc[df["model_r2-d"] == df["model_r2-d"].max()]
    drop = (max_pairs.loc[:, "model_r2-d"]/max_pairs.loc[:,"test_r2"]).min()
    if drop > threshold:
        return 1
    else: 
        return 0

def method_correct(df, strict, threshold):
    if strict:
        max_pairs = df.loc[df["model_r2-d"] == df["model_r2-d"].max()]
    else:
        max_pairs = df.loc[df["model_r2-d"] > df["model_r2-d"].max() - 0.05]
    
    correct = 0
    if df["ocsvm_score"].max() in max_pairs["ocsvm_score"].values:
        best_entry = max_pairs.loc[max_pairs["ocsvm_score"].idxmax()]
        drop = best_entry["model_r2-d"]/best_entry["test_r2"]
        if drop > threshold:
            correct = 1
    return correct

def precision(df, precision_args):
    correct = 0
    sample_ids = np.unique(df["sample"])
    for sample_id in sample_ids:
        sample_correct = 0
        sample_df = df.loc[(df["sample"] == sample_id)]
        similar_pairs = find_similar_pairs(sample_df)
        for x,y in similar_pairs[::2]:
            pair_df = sample_df.loc[((sample_df.model_node == x) & (sample_df.test_node == y))|
                                    ((sample_df.model_node ==y) & (sample_df.test_node == x))]
            if isinstance(precision_args, list):
                sample_correct += method_correct(pair_df, strict = precision_args[0], threshold = precision_args[1])
            elif isinstance(precision_args, bool):
                sample_correct += ocsvm_correct(pair_df, strict = precision_args)
            else:
                sample_correct += mmd_correct(pair_df, threshold = precision_args)
        correct += sample_correct/len(similar_pairs[::2])
    return round(correct/len(sample_ids),2)

def gnfuv_precision(df, precision_args):
    exp_precision = []
    for experiment in range(1,4):
        exp = df.loc[(df.experiment == experiment)]
        exp_precision.append(precision(exp, precision_args))
    weights = [df.loc[df.experiment == experiment].shape[0]/df.shape[0] for experiment in range(1,4)]
    weighted_avg_precision = sum([weights[i] * exp_precision[i] for i in range(3)])
    avg_precision = sum(exp_precision)/3
    return exp_precision, round(weighted_avg_precision,2), round(avg_precision,2)

# def banking_precision(df, precision_args):
#     return precision(df, precision_args)

def print_combined_precision(data, precision_fun):
    for threshold in [0.6, 0.8]:
        print(threshold, end=": ")
        for strict in [True, False]:
            print(precision_fun(data, [strict, threshold]), f"(strict={strict})", end = ", ")
        print()
    print()

def combined_precision(data, per_model_type = False): 
    if "kernel" in data.columns:
        print(normalized[data["std"].values[0]])
        precision_fun = gnfuv_precision
    else:
        precision_fun = precision
          
    if per_model_type:
        if "kernel" in data.columns:
            model_types = kernels
            attr = "kernel"
            print()
        else:
            model_types = balanced_options
            attr = "balanced"            
        for mt in model_types:
            d = data.loc[data[attr] == mt]
            print(f"{attr}={mt}")
            print_combined_precision(d, precision_fun)
    else:
        print_combined_precision(data, precision_fun) 

def print_ocsvm_precision(data, precision_fun):
    for strict in [True, False]:
        print(precision_fun(data, strict), f"(strict={strict})", end = ", ")
    print()

def ocsvm_precision(data, per_model_type=False):
    if "kernel" in data.columns:
        precision_fun = gnfuv_precision
    else:
        precision_fun = precision
    if per_model_type:
        if "kernel" in data.columns:
            model_types = kernels
            attr = "kernel"
        else:
            model_types = balanced_options
            attr = "balanced"            
        for mt in model_types:
            if "kernel" in data.columns: 
                print(normalized[data["std"].values[0]], end = ", ")
            d = data.loc[data[attr] == mt]
            print(f"{attr}={mt}:", end = " ")
            print_ocsvm_precision(data, precision_fun)
    else:
        if "kernel" in data.columns: 
            print(normalized[data["std"].values[0]], end = ": ")
        print_ocsvm_precision(data, precision_fun)

def print_mmd_precision(data, precision_fun):
    for threshold in [0.85, 0.9, 0.95]:                
        print(f"{threshold}: {precision_fun(data, threshold)}")
    print()
    
def mmd_precision(data, per_model_type=False):
    if "kernel" in data.columns:
        precision_fun = gnfuv_precision
        print(normalized[data["std"].values[0]])
    else:
        precision_fun = precision
    if per_model_type:
        if "kernel" in data.columns:
            model_types = kernels
            attr = "kernel"
            print()
        else:
            model_types = balanced_options
            attr = "balanced"            
        for mt in model_types:
            d = data.loc[data[attr] == mt]
            print(f"{attr}={mt}")
            print_mmd_precision(data, precision_fun)
    else:
        print_mmd_precision(data, precision_fun)

def get_saved_train_time(df):
    best_entry = df.loc[df["ocsvm_score"].idxmax()]
    reusable_node = best_entry["test_node"]
    mask = df["model_node"] == reusable_node
    train_time = sum(df.loc[mask,["train_time", "optimisation_time"]].values[0])
    return train_time, reusable_node

def speedup(df, nodes):
    saved_train_time = 0
    sample_ids = np.unique(df["sample"])
    for sample_id in sample_ids:
        sample_train_time = {}
        sample_node_train_time = {}
        sample_df = df.loc[(df["sample"] == sample_id)]
        similar_pairs = find_similar_pairs(sample_df)
        sample_nodes = np.unique(sample_df["model_node"])
        for node in sample_nodes:
            sample_node_train_time[node] = sum(sample_df.loc[sample_df.model_node == node, 
                                                             ["train_time", "optimisation_time"]].values[0])
        for x,y in similar_pairs[::2]:
            pair_df = sample_df.loc[((sample_df.model_node == x) & (sample_df.test_node == y))|
                                    ((sample_df.model_node ==y) & (sample_df.test_node == x))]
            stt, rn = get_saved_train_time(pair_df)
            if stt > 0: 
                sample_train_time[rn] = sample_node_train_time[rn]
        if len(sample_node_train_time) != 4:
            avg_train_time = sum(sample_node_train_time.values())/len(sample_node_train_time)
            for node in nodes:
                if node not in sample_node_train_time.keys():
                    sample_node_train_time[node] = avg_train_time
                    sample_train_time[node] = avg_train_time

        saved_train_time += 1- sum(sample_train_time.values())/sum(sample_node_train_time.values())
    return round(saved_train_time/len(sample_ids),2)

def gnfuv_speedup(df):
    exp_speedup = []
    for experiment in range(1,4):
        exp = df.loc[(df.experiment == experiment)]
        exp_speedup.append(speedup(exp, nodes = ["pi2", "pi3", "pi4", "pi5"]))
    weights = [df.loc[df.experiment == experiment].shape[0]/df.shape[0] for experiment in range(1,4)]
    weighted_avg_speedup = sum([weights[i] * exp_speedup[i] for i in range(3)])
    avg_speedup = sum(exp_speedup)/3
    return exp_speedup, round(weighted_avg_speedup,2), round(avg_speedup,2)

def banking_speedup(df, no_nodes):
    nodes = ["pi"+str(i+1) for i in range(no_nodes)]
    return speedup(df, nodes)