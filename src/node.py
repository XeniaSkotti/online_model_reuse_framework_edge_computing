import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

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

def standardise_node_data(experiment, node_data):
    scaler = StandardScaler()
    experiment_std_data = [pd.DataFrame(columns = ["humidity", "temperature"],
                                        data = scaler.fit_transform(node_data[i][["humidity", "temperature"]])) 
                           for i in range(4)]
    index=2
    for df in experiment_std_data:
        df["experiment"] = experiment
        df["pi"] = f"pi{index}"
        index+=1
    return experiment_std_data

def create_experiment_samples(n_samples, raw_data, standardised=False, experiment = False):
    if isinstance(experiment, bool):
        if experiment != False:
            raw_node_data = raw_data
    else:
        raw_node_data = get_node_data(raw_data, experiment, filtered=False)
        if standardised:
            raw_node_data = standardise_node_data(experiment,raw_node_data)
#     filtered_node_data, models = remove_outliers(raw_node_data, return_models = True)
    min_samples = min([d.shape[0] for d in raw_node_data])
    m = min_samples
    if min_samples/2 > 450:    
        m = int(min_samples/2)
    elif min_samples > 450:
        m = 450

    exp_samples = []
    no_nodes = len(np.unique(raw_data.pi))
    for sample_id in range(n_samples):
        node_data = raw_node_data.copy()
        for i in range(no_nodes):
            node_data[i] = raw_node_data[i].sample(m).reset_index(drop=True)
        exp_samples.append(node_data)
    return exp_samples

def create_samples(n_samples, raw_data, standardised=False): 
    if "experiment" in raw_data.columns:
        samples = {}
        ocsvm_data = {}
        for experiment in range(1,4):
            samples[experiment] = create_experiment_samples(n_samples, raw_data, standardised, experiment)
        return samples
    else:
        return create_experiment_samples(n_samples, raw_data)

def save_samples(samples, dataset, standardised = False):
    if "bank" in dataset:
        d = "bank-marketing"
    else:
        d = "GNFUV"
        if standardised:
            f = "standardised"
        else:
            f = "original"
        
    for sample_id in range(len(samples[1])):
        sample = []
        if "bank" not in dataset:
            for experiment in range(1,4):
                sample.append(pd.concat(samples[experiment][sample_id]))
            results = pd.concat(sample, ignore_index = True)
            directory = f"data/{d}/samples/{f}/sample_{sample_id+1}.csv"
        else:
            results = samples
            directory = f"data/{d}/samples/sample_{sample_id+1}.csv"
        results.to_csv(directory, index=False)