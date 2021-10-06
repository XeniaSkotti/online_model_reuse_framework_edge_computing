import numpy as np
import pandas as pd

def get_node_data(data, experiment):
    exp = data.loc[data.experiment==experiment][["humidity", "temperature"]]
    a = exp.loc[data.pi=="pi2"]
    b = exp.loc[data.pi=="pi3"]
    c = exp.loc[data.pi=="pi4"]
    d = exp.loc[data.pi =="pi5"]
    return remove_outliers([a,b,c,d], experiment)

def remove_outliers(node_data, experiment):
    if experiment == 1:
        for j in range(4):
            node_data[j] = node_data[j].loc[(node_data[j].humidity > 10) & (node_data[j].temperature > 10)]
    elif experiment == 2:
        for j in range(4):
            node_data[j] = node_data[j].loc[node_data[j].temperature > 32]
    return node_data