from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from prettytable import PrettyTable

def instantiate_clf(name):
    if name == "linear":
        return LinearRegression()
    elif name == "svr":
        return SVR()
    elif name == "lr":
        return LogisticRegression(max_iter = 10000, solver = "saga")

def fit_clf(clf, train, test):
    ## position 1 has temperature and position 0 humnidity
    clf.fit(train[:,1].reshape(-1,1), train[:,0])
    return score_clf(clf, test)

def score_clf(clf, test):
    return clf.score(test[:,1].reshape(-1,1), test[:,0])

def get_clf_param_grid(name):
    if name == "linear":
        param_grid = {"normalize" : [True, False]}
    elif name == "svr":
        param_grid = {"kernel": ["rbf", "linear"],
                      "C" : [0.05, 0.1, 0.5, 1, 2, 5, 10, 50],
                      "epsilon" : [0.05, 0.1, 0.15, 0.2, 0.5, 1]
                     }
    elif name == "lr":
        param_grid = {"penalty" : ["elasticnet"],
                      "C" : [0.05, 0.1, 0.5, 1, 2, 5, 10, 50],
                      "l1_ratio" : [0.1, 0.3, 0.5, 0.7, 0.9]
                     }
    return param_grid

def grid_search_models(clf_name, model_data, selected_nodes):
    models = {}
    t = PrettyTable(['Node', 'Baseline Model', 'Baseline Coefficient of Determination (R)',
                    'Optimised Model', 'Optimised Model R'])
    
    param_grid = get_clf_param_grid(clf_name)
    
    for node in selected_nodes:
        train, test = model_data[node] 
        
        baseline_model = instantiate_clf(clf_name)
        baseline_score = fit_clf(baseline_model, train, test)

        grid_search = GridSearchCV(instantiate_clf(clf_name), param_grid)
        grid_search.fit(train[:,1].reshape(-1,1), train[:,0])

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