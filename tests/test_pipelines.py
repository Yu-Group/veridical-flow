from functools import partial

import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils import resample

from vflow import ModuleSet, init_args  # must install vflow first (pip install vflow)
from vflow.pipeline import build_graph


class TestPipelines():

    def setup(self):
        pass

    def test_subsampling_fitting_metrics_pipeline(self):
        '''Simplest synthetic pipeline
        '''
        # initialize data
        np.random.seed(13)
        X, y = sklearn.datasets.make_classification(n_samples=50, n_features=5)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)  # ex. with another split?
        X_train, X_test, y_train, y_test = init_args((X_train, X_test, y_train, y_test),
                                                     names=['X_train', 'X_test', 'y_train',
                                                            'y_test'])  # optionally provide names for each of these

        # subsample data
        subsampling_funcs = [partial(sklearn.utils.resample,
                                     n_samples=20,
                                     random_state=i)
                             for i in range(3)]
        subsampling_set = ModuleSet(name='subsampling',
                                    modules=subsampling_funcs)
        X_trains, y_trains = subsampling_set(X_train, y_train)

        # fit models
        modeling_set = ModuleSet(name='modeling',
                                 modules=[LogisticRegression(max_iter=1000, tol=0.1),
                                          DecisionTreeClassifier()],
                                 module_keys=["LR", "DT"])

        modeling_set.fit(X_trains, y_trains)
        preds_test = modeling_set.predict(X_test)

        # get metrics
        hard_metrics_set = ModuleSet(name='hard_metrics',
                                     modules=[accuracy_score, balanced_accuracy_score],
                                     module_keys=["Acc", "Bal_Acc"])

        hard_metrics = hard_metrics_set.evaluate(preds_test, y_test)
        G = build_graph(hard_metrics, draw=True)

        # asserts
        k1 = (('X_train', 'y_train', 'subsampling_0', 'LR', 'X_test'),
              'y_test', 'Acc')
        assert k1 in hard_metrics, 'hard metrics should have ' + str(k1) + ' as key'
        assert hard_metrics[k1] > 0.9  # 0.9090909090909091
        assert '__prev__' in hard_metrics
        assert len(hard_metrics.keys()) == 13

    def test_feat_engineering(self):
        '''Feature engineering pipeline
        '''
        # get data as df
        np.random.seed(13)
        data = sklearn.datasets.load_boston()
        df = pd.DataFrame.from_dict(data['data'])
        df.columns = data['feature_names']
        y = data['target']
        X_train, X_test, y_train, y_test = init_args(train_test_split(df, y, random_state=123),
                                                     names=['X_train', 'X_test', 'y_train', 'y_test'])

        # feature extraction - extracts two different sets of features from the same data
        def extract_feats(df: pd.DataFrame, feat_names=['CRIM', 'ZN', 'INDUS', 'CHAS']):
            '''extract specific columns from dataframe
            '''
            return df[feat_names]

        feat_extraction_funcs = [partial(extract_feats, feat_names=['CRIM', 'ZN', 'INDUS', 'CHAS']),
                                 partial(extract_feats, feat_names=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE']),
                                 ]
        feat_extraction = ModuleSet(name='feat_extraction',
                                    modules=feat_extraction_funcs)

        X_feats_train = feat_extraction(X_train)
        X_feats_test = feat_extraction(X_test)

        modeling_set = ModuleSet(name='modeling',
                                 modules=[DecisionTreeRegressor(), RandomForestRegressor()],
                                 module_keys=["DT", "RF"])

        # how can we properly pass a y here so that it will fit properly?
        # this runs, but modeling_set.out is empty
        _ = modeling_set.fit(X_feats_train, y_train)

        # #get predictions
        preds_all = modeling_set.predict(X_feats_train)

        # y_test_dict = {('data_0', 'feat_extraction_0'): y_test['X_test'], ('data_0', 'feat_extraction_1'): y_test['X_test']}

        # get metrics
        hard_metrics_set = ModuleSet(name='hard_metrics',
                                     modules=[r2_score],
                                     module_keys=["r2"])
        hard_metrics = hard_metrics_set.evaluate(preds_all, y_train)

        # inspect the pipeline
        # for k in hard_metrics:
        #     print(k, hard_metrics[k])
        G = build_graph(hard_metrics, draw=True)

        # asserts
        k1 = (((('X_train', 'feat_extraction_0'), 'y_train', 'DT'),
               ('X_train', 'feat_extraction_0')),
              'y_train', 'r2')
        assert k1 in hard_metrics, 'hard metrics should have ' + str(k1) + ' as key'
        assert hard_metrics[k1] > 0.9  # 0.9090909090909091
        assert '__prev__' in hard_metrics
        assert len(hard_metrics.keys()) == 5

    def test_feature_importance(self):
        '''Simplest synthetic pipeline for feature importance
        '''
        # initialize data
        np.random.seed(13)
        X, y = sklearn.datasets.make_classification(n_samples=50, n_features=5)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)  # ex. with another split?
        X_train, X_test, y_train, y_test = init_args((X_train, X_test, y_train, y_test),
                                                     names=['X_train', 'X_test', 'y_train',
                                                            'y_test'])  # optionally provide names for each of these

        # subsample data
        subsampling_funcs = [partial(sklearn.utils.resample,
                                     n_samples=20,
                                     random_state=i)
                             for i in range(3)]
        subsampling_set = ModuleSet(name='subsampling',
                                    modules=subsampling_funcs)
        X_trains, y_trains = subsampling_set(X_train, y_train)

        # fit models
        modeling_set = ModuleSet(name='modeling',
                                 modules=[LogisticRegression(max_iter=1000, tol=0.1),
                                          DecisionTreeClassifier()],
                                 module_keys=["LR", "DT"])

        modeling_set.fit(X_trains, y_trains)
        preds_test = modeling_set.predict(X_test)

        # get metrics
        feature_importance_set = ModuleSet(name='feature_importance', modules=[permutation_importance],
                                           module_keys=["permutation_importance"])
        importances = feature_importance_set.evaluate(modeling_set.out, X_test, y_test)

        # asserts
        k1 = (('X_train', 'y_train', 'subsampling_0', 'LR'), 'X_test', 'y_test', 'permutation_importance')
        assert k1 in importances, 'hard metrics should have ' + str(k1) + ' as key'
        assert '__prev__' in importances
        assert len(importances.keys()) == 7
