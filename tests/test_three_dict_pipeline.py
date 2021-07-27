from pcsp.convert import combine_three_subset_dicts, extend_dicts, full_combine_two_dicts
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

class TestThreeDictPipeline():
    
    def setup(self):
        pass

    def test_extend_dict(self):
        arg_names = ['x_train', 'y_train', 'classifiers']
        vals_tuple = ('x_train', 'y_train', ['log_reg'])
        x_train, y_train, classifiers = init_args(vals_tuple, names=arg_names)
        combined_dict = extend_dicts(x_train, y_train, classifiers)
        for arg in arg_names:
            assert arg in combined_dict.keys(), 'dict should contain ' + str(arg) + ' as key'
        for val in vals_tuple:
            assert val in combined_dict.values(), 'dict should contain ' + str(val) + ' as val'
    
    def test_full_combine_dicts(self):
        x_train, y_train = init_args(('x_train', 'y_train'), names=['x_train', 'y_train'])
        merged_dict = extend_dicts(x_train, y_train)
        non_merged = {('classifier 1', 'rf'): lambda x: x, ('classifier 2', 'dt'): lambda x: x}
        combined_dict = full_combine_two_dicts(merged_dict, non_merged)
        k1 = (('classifier 1', 'rf'), 'x_train', 'y_train')
        k2 = (('classifier 2', 'dt'), 'x_train', 'y_train')
        for k in [k1, k2]:
            assert k in combined_dict.keys(), 'dict should contain ' + str(k) + ' as key'
    
    def test_combine_three_dicts(self):
        x_train, y_train = init_args(('x_train', 'y_train'), names=['x_train', 'y_train'])
        classifiers = {('classifier 1', 'rf'): 0, ('classifier 2', 'dt'): 1}
        combined_dict = combine_three_subset_dicts(x_train, y_train, classifiers)
        k1 = (('classifier 1', 'rf'), 'x_train', 'y_train')
        k2 = (('classifier 2', 'dt'), 'x_train', 'y_train')
        for k in [k1, k2]:
            assert k in combined_dict.keys(), 'dict should contain ' + str(k) + ' as key'
        v1 = [0, 'x_train', 'y_train']
        v2 = [1, 'x_train', 'y_train']
        for v in [v1, v2]:
            assert v in combined_dict.values(), 'dict should contain ' + str(v) + ' as value'
    
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

        # asserts
        k1 = (('X_train', 'y_train', 'subsampling_0', 'LR'), 'X_test', 'y_test', 'permutation_importance')
        assert k1 in importances, 'hard metrics should have ' + str(k1) + ' as key'
        