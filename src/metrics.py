import pandas as pd
import numpy as np 

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
        sample_df = df.loc[(df["sample"] == sample)]
        similar_pairs = find_similar_pairs(sample_df)
        for x,y in similar_pairs[::2]:
            pair_df = exp.loc[((exp.model_node == x) & (exp.test_node == y))|((exp.model_node ==y) & (exp.test_node == x))]
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

def banking_precision(df, precision_args):
    return precision(df, precision_args)