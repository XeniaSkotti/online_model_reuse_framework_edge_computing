from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR, LinearSVR
from sklearn.model_selection import GridSearchCV
from prettytable import PrettyTable

def instantiate_clf(name):
    if name == "lreg":
        return LinearRegression()
    elif name == "svr":
        return SVR()
    elif name == "lsvr":
        return SVR(kernel="linear")
    
def fit_clf(clf, train, test):
    ## position 1 has temperature and position 0 humnidity
    clf.fit(train[0].reshape(-1,1), train[1])
    return score_clf(clf, test)

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
    t = PrettyTable(['Node', 'Baseline Model', 'Baseline R',
                    'Optimised Model', 'Optimised Model R'])
    
    param_grid = get_clf_param_grid(clf_name)
    
    for node in selected_nodes:
        train, test = model_data[node]
        
        baseline_model = instantiate_clf(clf_name)
        baseline_score = fit_clf(baseline_model, train, test)

        grid_search = GridSearchCV(instantiate_clf(clf_name), param_grid)
        grid_search.fit(train[0].reshape(-1,1), train[1])

        optimised_model = instantiate_clf(clf_name)
        optimised_model.set_params(**grid_search.best_params_)
        optimised_score = fit_clf(optimised_model, train, test)
        t.add_row([node, baseline_model, baseline_score, optimised_model, optimised_score])
        
        if optimised_score > baseline_score:
            model = optimised_model
        else:
            model = baseline_model
            
        models[node] = model
    
    print(t)
    
    return models