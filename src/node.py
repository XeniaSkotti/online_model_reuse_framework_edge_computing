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

def remove_outliers(node_data, return_models=False):
    if return_models:
        models = []
        
    for i in range(4):
        model = OneClassSVM(nu=0.1)
        model.fit(node_data[i][["humidity", "temperature"]])
        if return_models:
            models.append(model)
        
    if return_models:
        return node_data, models
    else:
        return node_data
    
def create_samples(raw_data, standardised): 
    ocsvm_data = {}
    samples = {} 
    for experiment in range(1,4):
        raw_node_data = get_node_data(raw_data, experiment, filtered=False)
        if standardised:
            raw_node_data = standardise_node_data(experiment,raw_node_data)
        
        filtered_node_data, models = remove_outliers(raw_node_data, return_models = True)
        min_samples = min([d.shape[0] for d in filtered_node_data])
        m = min_samples
        if min_samples/2 > 450:    
            m = int(min_samples/2)
        elif min_samples > 450:
            m = 450
            
        exp_samples = []
        for sample_id in range(4):
            node_data = filtered_node_data.copy()
            for i in range(4):
                node_data[i] = filtered_node_data[i].sample(m).reset_index(drop=True)
            exp_samples.append(node_data)
        samples[experiment] = exp_samples
        ocsvm_data[experiment] = {"models" : models, "raw_node_data": raw_node_data}
    return ocsvm_data, samples