import pandas as pd
import numpy as np

kernels = ["rbf", "linear"]
balanced_options = [True, False]

def find_similar_pairs(df):
    unique_pairs_df = pd.DataFrame.from_records(np.unique(df[["model_node", "test_node"]].to_records(index=False))) 
    similar_pairs = sort_similar_pairs(unique_pairs_df)
    return similar_pairs

def sort_similar_pairs(df):
    n_pairs =  int(df.shape[0]/2)
    
    for pair_index in range(n_pairs):
        pair = df.iloc[pair_index*2].copy()
        backwards_pair =  df.loc[(df.test_node == pair.model_node) & 
                                (df.model_node == pair.test_node)]

        if pair_index*2+1 != backwards_pair.index:
            misplaced_pair = df.loc[pair_index*2+1]
            df.iloc[backwards_pair.index] = misplaced_pair
            df.iloc[pair_index*2+1] = backwards_pair

        if pair.model_node > pair.test_node:
            df.iloc[pair_index*2] = backwards_pair
            df.iloc[pair_index*2+1] = pair

    similar_pairs = [pair for pair in df.values]
    
    return similar_pairs

def get_pair_df(pair, df):
    forward_match = ((df.model_node==pair[0])&(df.test_node==pair[1]))
    backward_match = ((df.model_node==pair[1])&(df.test_node==pair[0])) 
    forward_df = df.loc[forward_match]
    backward_df = df.loc[backward_match]
    return {str((pair[0], pair[1])) : forward_df, str((pair[1], pair[0])) : backward_df}

def merge_data(df):
    similar_pairs = find_similar_pairs(df)
    merged = []
    if "kernel" in df.columns:
        model_types = kernels
        attr = "kernel"
    else:
        model_types = balanced_options
        attr = "balanced"
    for x,y in similar_pairs:
        for mt in model_types:
            pair_df = df.loc[(df.model_node == x) & (df.test_node==y) & (df[attr] == mt)]
            mean_data = pair_df.mean().round(2)
            merged_pair_df = pd.DataFrame(columns=mean_data.index.values)
            merged_pair_df.loc[0] = mean_data.values
            merged_pair_df.round(1)
            merged_pair_df["model_node"] = x
            merged_pair_df["test_node"] = y
            merged_pair_df[attr] = mt
            merged.append(merged_pair_df)
    return pd.concat(merged, ignore_index = True)

def merge_gnfuv_results(data):
    merged_data = []
    for dataset in data:
        experiment_merged_dfs = []
        for exp in range(1,4):
            df = dataset.loc[(dataset.experiment == exp)]
            merged_exp_df = merge_data(df)
            merged_exp_df["std"] = df["std"].values[0]
            merged_exp_df["experiment"] = exp
            experiment_merged_dfs.append(merged_exp_df)
        merged_data.append(pd.concat(experiment_merged_dfs, ignore_index=True))
    return merged_data

def merge_banking_results(data):
    return merge_data(data)