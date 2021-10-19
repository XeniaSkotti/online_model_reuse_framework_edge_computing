from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR, LinearSVR
import pandas as pd


def get_xy_data(data):
    x = data.humidity.values.astype(np.float32)
    y = data.temperature.values.astype(np.float32)
    return x, y

def get_model_data(node_data, standardised = False):
    model_data = {}
    
    if standardised:
        scaler = StandardScaler()
        scaler.fit(pd.concat(node_data)[["humidity", "temperature"]])
    
    for i in range(4):
        node = "pi"+str(i+2)
        if standardised:
            std_node = scaler.transform(node_data[i][["humidity", "temperature"]])
            model_data[node] = (std_node[:,0],std_node[:,1])
        else:
            model_data[node] = get_xy_data(node_data[i])
        
    return model_data

def select_model_data(node_data, similar_nodes, standardised):
    model_data = get_model_data(node_data, standardised)
    selected_model_data = {node : model_data[node] for node in similar_nodes}
    return selected_model_data

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
        param_grid = {"C" : [0.00001, 0.0001, 0.001, 0.01, 0.1,  1, 10, 100],
                      "epsilon" : [0.01, 0.1, 0.1, 0.2, 0.5, 1]
                     }
    elif name == "lsvr":
        param_grid = {"C" : [0.00001, 0.0001, 0.001, 0.01, 0.1,  1, 10, 100],
                      "epsilon" : [0, 0.01, 0.1, 0.2, 0.5, 1],
                     }
    return param_grid

def grid_search_models(clf_name, model_data, selected_nodes):
    models = {}
    l = []
    
    param_grid = get_clf_param_grid(clf_name)
    
    for node in selected_nodes:
        train = model_data[node]
        
        baseline_model = instantiate_clf(clf_name)
        baseline_score = fit_clf(baseline_model, train)

        grid_search = GridSearchCV(instantiate_clf(clf_name), param_grid)
        grid_search.fit(train[0].reshape(-1,1), train[1])

        optimised_model = instantiate_clf(clf_name)
        optimised_model.set_params(**grid_search.best_params_)
        optimised_score = fit_clf(optimised_model, train)
        
        if optimised_score > baseline_score:
            model = optimised_model
            score = optimised_score
        else:
            model = baseline_model
            score = baseline_score
            
        models[node] = model
        l.append(pd.DataFrame([{"model_node" :  node, "model" : model, "score" : score}]))
    
    return models, pd.concat(l)