from modelling import grid_search_models, fit_clf, score_clf
import pandas as pd

def test_in_pairs(similar_pairs, model_data, models, mmd_scores, ocsvm_scores):
    l = []
    for i in range(len(similar_pairs)):
        node_x, node_y = similar_pairs[i]
        x = model_data[node_x]
        y = model_data[node_y]
        
        model_x = models[node_x]
        ex = fit_clf(model_x, x)
        exy = score_clf(model_x, y)

        model_y = models[node_y]
        ey = fit_clf(model_y, y)
        eyx = score_clf(model_y, x)       
        
        l.append(pd.DataFrame([{"model_node" : node_x, "test_node" : node_y, "discrepancy" : round(abs(ex-exy),2), 
                                "model_r2-d" : exy, "test_r2" : ey, "mmd_score" : mmd_scores[i], "ocsvm_score" : ocsvm_scores[i][0]}]))
        l.append(pd.DataFrame([{"model_node" : node_y, "test_node" : node_x, "discrepancy" : round(abs(ey-eyx),2),
                                "model_r2-d" : eyx, "test_r2" : ex, "mmd_score" : mmd_scores[i], "ocsvm_score" : ocsvm_scores[i][1]}]))
    return pd.concat(l, ignore_index = True)

def test_hypothesis(clf_name, model_data, similar_pairs, similar_nodes, mmd_scores, ocsvm_scores): 
    models, models_df = grid_search_models(clf_name, model_data, similar_nodes)
    test_df = test_in_pairs(similar_pairs, model_data, models, mmd_scores, ocsvm_scores)
    return models_df.merge(test_df, how='outer', on='model_node')