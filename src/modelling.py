from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from prettytable import PrettyTable

def instantiate_clf(name):
    if name == "linear":
        return LinearRegression()
    elif name == "svr":
        return SVR()

def fit_clf(clf, train, test):
    ## position 1 has temperature and position 0 humnidity
    clf.fit(train[:,1].reshape(-1,1), train[:,0])
    return score_clf(clf, test)

def score_clf(clf, test):
    return clf.score(test[:,1].reshape(-1,1), test[:,0])

def grid_search_models(clf_name, model_data, selected_nodes, param_grid):
    models = {}
    t = PrettyTable(['Node', 'Baseline Model', 'Baseline Coefficient of Determination (R)',
                    'Optimised Model', 'Optimised Model Coeffficient of Determination (R'])
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