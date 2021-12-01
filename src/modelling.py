from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR
import pandas as pd
import numpy as np
import time

def get_gnfuv_xy_data(data):
    x = data.humidity.values.astype(np.float32)
    y = data.temperature.values.astype(np.float32)
    return x, y

def get_banking_xy_data(data):
    x = data[["x", "y", "z"]].values.astype(np.float32)
    y = data.label.values
    return x,y

def select_model_data(node_data, similar_nodes):
    model_data = {}
    for i in range(len(node_data)):
        n = node_data[i]
        node = n.pi.values[0]
        if node in similar_nodes:
            if "experiment" in n.columns:
                model_data[node] = get_gnfuv_xy_data(n)
            else:
                model_data[node] = get_banking_xy_data(n)
    return model_data

def instantiate_clf(name):
    if name == "lr":
        return LogisticRegression(max_iter = 100000)
    elif name == "svr":
        return SVR()
    elif name == "lsvr":
        return SVR(kernel="linear")
    
def fit_clf(clf, train):
    ## position 1 has temperature and position 0 humnidity
    if train[0].shape[1] == 1:
        clf.fit(train[0].reshape(-1,1), train[1])
    else:
         clf.fit(train[0], train[1])
    return score_clf(clf, train)

def score_clf(clf, test):
    if test[0].shape[1] == 1:
        score = clf.score(test[0].reshape(-1,1), test[1])
    else:
        score = clf.score(test[0], test[1])
    return score

def get_clf_param_grid(name):
    if name == "lr":
        param_grid = {"C" :  [0.01, 0.1,  1, 10],
                      "solver" : ["lbfgs","liblinear", "saga", "sag"],
                     }
    elif name == "svr":
        param_grid = {"C" : [0.01, 0.1,  1, 10],
                      "epsilon" : [0.1, 0.5, 1, 2, 5],
                     }
    elif name == "lsvr":
        param_grid = {"C" : [0.01, 0.1,  1, 10],
                      "epsilon" : [0.1, 0.5, 1, 2, 5],
                     }
    return param_grid

def grid_search_models(clf_name, model_data, selected_nodes):
    models = {}
    l = []
    
    param_grid = get_clf_param_grid(clf_name)
    
    for node in selected_nodes:
        train = model_data[node]
        
        baseline_model = instantiate_clf(clf_name)
        start_time = time.time()
        baseline_score = fit_clf(baseline_model, train)
        baseline_train_time = time.time() - start_time
        
        grid_search = GridSearchCV(instantiate_clf(clf_name), param_grid)
        start_time = time.time()
        if train[0].shape[1] == 1:
            grid_search.fit(train[0].reshape(-1,1), train[1])
        else:
            grid_search.fit(train[0], train[1])
        optimisation_time = time.time() - start_time

        optimised_model = instantiate_clf(clf_name)
        optimised_model.set_params(**grid_search.best_params_)
        start_time = time.time()
        optimised_score = fit_clf(optimised_model, train)
        optimised_train_time = time.time() - start_time
        
        if optimised_score > baseline_score:
            model = optimised_model
            score = optimised_score
            train_time = optimised_train_time
        else:
            model = baseline_model
            score = baseline_score
            train_time = baseline_train_time
            
        models[node] = model
        l.append(pd.DataFrame([{"model_node" :  node, "model" : model, "train_time" : round(train_time,2), 
                                "optimisation_time" : round(optimisation_time, 2), "model_r2" : round(score,2)}]))
    
    return models, pd.concat(l)