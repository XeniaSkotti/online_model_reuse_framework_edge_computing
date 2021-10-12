from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

def instantiate_clf(name):
    if name == "linear":
        return LinearRegression()
    elif name == "svr"
        return SVR()

def fit_clf(clf, train, test):
    ## position 1 has temperature and position 0 humnidity
    clf.fit(train[:,1].reshape(-1,1), train[:,0])
    return score_clf(clf, test)

def score_clf(clf, test):
    return clf.score(test[:,1].reshape(-1,1), test[:,0])

def grid_search_models(clf_name, model_data, selected_nodes, param_grid):
    models = {}
    for node in selected_nodes:
        train, test = model_data[node] 
        
        model = instantiate_clf(clf_name)
        score = fit_clf(model, train, test)
        print(f"Baseline Model Coeefficient of Determination for node {node}: {score}")

        grid_search = GridSearchCV(instantiate_clf(clf_name), param_grid)
        grid_search.fit(train[:,1].reshape(-1,1), train[:,0])
        best_params = grid_search.best_params_

        print(f"The best parameter settings for node {node} are: {best_params}")
        model = instantiate_clf(clf_name)
        model.set_params(**best_params)
        score = fit_clf(model, train, test)
        print(f"Optimised Model Coeefficient of Determination for node {node}: {score} \n")
        
        models[node] = model

    return models