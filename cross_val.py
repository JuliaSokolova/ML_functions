from sklearn.model_selection import KFold
import numpy as np

def cross_val(X, y, my_model, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True)  # almost always use shuffle=True
    fold_scores = []

    for train, test in kf.split(X):
        mod = my_model()
        mod.fit(X.values[train], y.values[train])
        fold_scores.append(mod.score(X.values[test], y.values[test]))
        
    return(np.mean(fold_scores))