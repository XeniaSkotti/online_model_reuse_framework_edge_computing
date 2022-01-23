import pandas as pd
import numpy as np

kernels = ["rbf", "linear"]
balanced_options = [True, False]
balanced_text = ["balanced", "unbalanced"]

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

    similar_pairs = [(x,y) for x,y in df.values]
    
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
            if not pair_df.empty:
                mean_data = pair_df.mean().round(2)
                merged_pair_df = pd.DataFrame(columns=mean_data.index.values)
                merged_pair_df.loc[0] = mean_data.values
                merged_pair_df.round(1)
                merged_pair_df["model_node"] = x
                merged_pair_df["test_node"] = y
                merged_pair_df[attr] = mt
                merged.append(merged_pair_df.drop(columns = ["sample"]))
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

def find_the_best_entry_per_sample(df):
    results = pd.DataFrame()
    sample_ids = np.unique(df["sample"])
    for sample_id in sample_ids:
        sample_df = df.loc[(df["sample"] == sample_id)]
        similar_pairs = find_similar_pairs(sample_df)
        for x,y in similar_pairs[::2]:
            pair_df = sample_df.loc[((sample_df.model_node == x) & (sample_df.test_node == y))|
                                    ((sample_df.model_node == y) & (sample_df.test_node == x))]
            max_entry = pair_df.loc[pair_df["model_r2-d"].idxmax()]
            results = pd.concat([results, max_entry.to_frame().T])
    return results

# def find_the_best_entry_per_sample_gnfuv(df):
#     results = []
#     for experiment in range(1,4):
#         exp = df.loc[(df.experiment == experiment)]
#         results.append(find_the_best_entry_per_sample(exp))
#     return pd.concat(results)

def gnfuv_data_summary(df, best_entry= False):
     for experiment in range(1,4):
        exp = df.loc[(df.experiment == experiment)]
        print(f"Data sunmary for experiment {experiment}")
        data_summary(exp, best_entry)
        print()
        
def data_summary(df, best_entry = False):
    if "kernel" in df.columns:
        model_types = kernels
        attr = "kernel"
        text = kernels
    else:
        model_types = balanced_options
        attr = "balanced" 
        text = balanced_text
    if best_entry:
        df = find_the_best_entry_per_sample(df)
        
    type_1_df = df.loc[(df[attr] == model_types[0])]
    type_2_df = df.loc[(df[attr] == model_types[1])]
    
    if best_entry:
        print(f"The number of entries per model type are {text[0]}={type_1_df.shape[0]}, {text[1]}={type_2_df.shape[0]}")
    if min(type_1_df.shape[0], type_2_df.shape[0]) == 0:
        if type_1_df.shape[0] < type_2_df.shape[0]:
            baseline_score, lower_discrepancy, best_model_type = text[1], text[1], text[1]
        else:
            baseline_score, lower_discrepancy, best_model_type = text[0], text[0], text[0]
    else:
        if type_1_df.model_r2.mean() > type_2_df.model_r2.mean():
            baseline_score = text[0]
        else:
            baseline_score = text[1]
        if type_1_df.discrepancy.mean() > type_2_df.discrepancy.mean():
            lower_discrepancy = text[1]
        else:
            lower_discrepancy = text[0]
        if type_1_df["model_r2-d"].mean() > type_2_df["model_r2-d"].mean():
            best_model_type = text[0]
        else:
            best_model_type = text[1]
    print(f"{baseline_score} models have higher baseline R2 scores", end = " and ")
    print(f"{lower_discrepancy} models have lower discrepancy", end = ". \n")
    print(f"{best_model_type} models yield the best results on average. \n")