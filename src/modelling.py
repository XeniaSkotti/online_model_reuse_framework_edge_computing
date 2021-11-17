from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR, LinearSVR
import pandas as pd
import numpy as np
import time

def get_xy_data(data):
    x = data.humidity.values.astype(np.float32)
    y = data.temperature.values.astype(np.float32)
    return x, y

def select_model_data(node_data, similar_nodes):
    model_data = {}
    for i in range(4):
        node = "pi"+str(i+2)
        if node in similar_nodes:
            model_data[node] = get_xy_data(node_data[i])
    return model_data

def instantiate_clf(name):
    if name == "lreg":
        return LinearRegression()
    elif name == "svr":
        return SVR()
    elif name == "lsvr":
        return SVR(kernel="linear")
    
def fit_clf(clf, train):
    ## position 1 has temperature and position 0 humnidity
    clf.fit(train[0].reshape(-1,1), train[1])
    return score_clf(clf, train)

def score_clf(clf, test):
    return clf.score(test[0].reshape(-1,1), test[1])

def get_clf_param_grid(name):
    if name == "lreg":
        param_grid = {"normalize" : [True, False]}
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
        grid_search.fit(train[0].reshape(-1,1), train[1])
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