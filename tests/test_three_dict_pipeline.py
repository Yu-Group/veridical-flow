import pcsp
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from functools import partial
from pcsp import PCSPipeline, ModuleSet, Module, init_args # must install pcsp first (pip install pcsp)
from pcsp.pipeline import build_graph
import sklearn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
import pandas as pd

class TestPipelines():
    
    def setup(self):
        pass
    
    def test_feature_importance(self):
        '''Simplest synthetic pipeline for feature importance
        '''
        # initialize data
        np.random.seed(13)
        X, y = sklearn.datasets.make_classification(n_samples=50, n_features=5)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42) # ex. with another split?
        X_train, X_test, y_train, y_test = init_args((X_train, X_test, y_train, y_test),
                                                      names=['X_train', 'X_test', 'y_train', 'y_test'])  # optionally provide names for each of these

        # subsample data
        subsampling_funcs = [partial(sklearn.utils.resample,
                                    n_samples=20,
                                    random_state=i)
                             for i in range(3)]
        subsampling_set = ModuleSet(name='subsampling',
                                    modules=subsampling_funcs)
        X_trains, y_trains = subsampling_set(X_train, y_train)


        #fit models
        modeling_set = ModuleSet(name='modeling',
                                  modules=[LogisticRegression(max_iter=1000, tol=0.1),
                                           DecisionTreeClassifier()],
                                  module_keys=["LR", "DT"])

        modeling_set.fit(X_trains, y_trains)
        preds_test = modeling_set.predict(X_test)

        # get metrics
        feature_importance_set = ModuleSet(name='feature_importance', modules=[permutation_importance], module_keys=["permutation_importance"])
        importances = feature_importance_set.evaluate(modeling_set.out, X_test, y_test)

        G = build_graph(importances, draw=True)
        
        print(importances.keys())
        # asserts
        k1 = (('X_train', 'y_train', 'subsampling_0', 'LR'), 'X_test', 'y_test', 'permutation_importance')
        assert k1 in importances, 'hard metrics should have ' + str(k1) + ' as key'
        