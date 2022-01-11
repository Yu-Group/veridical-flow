import time
import os
from functools import partial
from shutil import rmtree

import numpy as np
import pandas as pd
import ray
import sklearn
from numpy.testing import assert_equal
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from vflow import Vset, init_args, build_vset  # must install vflow first (pip install vflow)
from vflow.subkey import Subkey as sm
from vflow.vset import PREV_KEY


class TestPipelines:

    def setup(self):
        pass

    def test_subsampling_fitting_metrics_pipeline(self):
        """Simplest synthetic pipeline
        """
        # initialize data
        np.random.seed(13)
        X, y = sklearn.datasets.make_classification(n_samples=50, n_features=5)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)  # ex. with another split?
        X_train, X_test, y_train, y_test = init_args((X_train, X_test, y_train, y_test),
                                                     names=['X_train', 'X_test', 'y_train',
                                                            'y_test'])  # optionally provide names for each of these

        # subsample data
        subsampling_set = build_vset('subsampling', sklearn.utils.resample,
                                     param_dict={'random_state': list(range(3))},
                                     n_samples=20, verbose=False)
        X_trains, y_trains = subsampling_set(X_train, y_train)

        # fit models
        modeling_set = Vset(name='modeling',
                            modules=[LogisticRegression(max_iter=1000, tol=0.1),
                                     DecisionTreeClassifier()],
                            module_keys=["LR", "DT"])

        modeling_set.fit(X_trains, y_trains)

        preds_test = modeling_set.predict(X_test)

        # get metrics
        hard_metrics_set = Vset(name='hard_metrics',
                                modules=[accuracy_score, balanced_accuracy_score],
                                module_keys=["Acc", "Bal_Acc"])

        hard_metrics = hard_metrics_set.evaluate(preds_test, y_test)

        # asserts
        k1 = (sm('X_test', 'init'), sm('X_train', 'init'), sm('subsampling_0', 'subsampling'),
              sm('y_train', 'init'), sm('LR', 'modeling'), sm('y_test', 'init'), sm('Acc', 'hard_metrics'))

        assert k1 in hard_metrics, 'hard metrics should have ' + str(k1) + ' as key'
        assert hard_metrics[k1] > 0.9  # 0.9090909090909091
        assert PREV_KEY in hard_metrics
        assert len(hard_metrics.keys()) == 13

    def test_feat_engineering(self):
        """Feature engineering pipeline
        """
        # get data as df
        np.random.seed(13)
        data = sklearn.datasets.load_boston()
        df = pd.DataFrame.from_dict(data['data'])
        df.columns = data['feature_names']
        y = data['target']
        X_train, X_test, y_train, y_test = init_args(train_test_split(df, y, random_state=123),
                                                     names=['X_train', 'X_test', 'y_train', 'y_test'])

        # feature extraction - extracts two different sets of features from the same data
        def extract_feats(df: pd.DataFrame, feat_names=None):
            """extract specific columns from dataframe
            """
            if feat_names is None:
                feat_names = ['CRIM', 'ZN', 'INDUS', 'CHAS']
            return df[feat_names]

        feat_extraction_funcs = [partial(extract_feats, feat_names=['CRIM', 'ZN', 'INDUS', 'CHAS']),
                                 partial(extract_feats, feat_names=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE']),
                                 ]
        feat_extraction = Vset(name='feat_extraction',
                               modules=feat_extraction_funcs,
                               output_matching=True)

        X_feats_train = feat_extraction(X_train)

        modeling_set = Vset(name='modeling',
                            modules=[DecisionTreeRegressor(), RandomForestRegressor()],
                            module_keys=["DT", "RF"])

        # how can we properly pass a y here so that it will fit properly?
        # this runs, but modeling_set.out is empty
        _ = modeling_set.fit(X_feats_train, y_train)

        # #get predictions
        preds_all = modeling_set.predict(X_feats_train)

        # get metrics
        hard_metrics_set = Vset(name='hard_metrics',
                                modules=[r2_score],
                                module_keys=["r2"])
        hard_metrics = hard_metrics_set.evaluate(preds_all, y_train)

        # asserts
        k1 = (sm('X_train', 'init'), sm('feat_extraction_0', 'feat_extraction', True), sm('X_train', 'init'),
              sm('y_train', 'init'),
              sm('DT', 'modeling'), sm('y_train', 'init'), sm('r2', 'hard_metrics'))
        assert k1 in hard_metrics, 'hard metrics should have ' + str(k1) + ' as key'
        assert hard_metrics[k1] > 0.9  # 0.9090909090909091
        assert PREV_KEY in hard_metrics
        assert len(hard_metrics.keys()) == 5

    def test_feature_importance(self):
        """Simplest synthetic pipeline for feature importance
        """
        # initialize data
        np.random.seed(13)
        X, y = sklearn.datasets.make_classification(n_samples=50, n_features=5)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)  # ex. with another split?
        X_train, X_test, y_train, y_test = init_args((X_train, X_test, y_train, y_test),
                                                     names=['X_train', 'X_test', 'y_train',
                                                            'y_test'])  # optionally provide names for each of these

        # subsample data
        subsampling_set = build_vset('subsampling', sklearn.utils.resample,
                                     param_dict={'random_state': list(range(3))},
                                     n_samples=20, verbose=False)
        X_trains, y_trains = subsampling_set(X_train, y_train)

        # fit models
        modeling_set = Vset(name='modeling',
                            modules=[LogisticRegression(max_iter=1000, tol=0.1),
                                     DecisionTreeClassifier()],
                            module_keys=["LR", "DT"])

        modeling_set.fit(X_trains, y_trains)
        preds_test = modeling_set.predict(X_test)

        # get metrics
        feature_importance_set = Vset(name='feature_importance', modules=[permutation_importance],
                                      module_keys=["permutation_importance"])
        importances = feature_importance_set.evaluate(modeling_set.out, X_test, y_test)

        # asserts
        k1 = (sm('X_train', 'init'), sm('subsampling_0', 'subsampling'),
              sm('y_train', 'init'), sm('LR', 'modeling'), sm('X_test', 'init'),
              sm('y_test', 'init'), sm('permutation_importance', 'feature_importance'))
        assert k1 in importances, 'hard metrics should have ' + str(k1) + ' as key'
        assert PREV_KEY in importances
        assert len(importances.keys()) == 7

    def test_repeated_subsampling(self):
        np.random.seed(13)
        X, y = sklearn.datasets.make_classification(n_samples=50, n_features=5)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
        X_train, X_test, y_train, y_test = init_args((X_train, X_test, y_train, y_test),
                                                     names=['X_train', 'X_test', 'y_train', 'y_test'])

        # subsample data
        subsampling_set = build_vset('subsampling', sklearn.utils.resample,
                                     param_dict={'random_state': list(range(3))},
                                     n_samples=20)
        X_trains, y_trains = subsampling_set(X_train, y_train)
        X_tests, y_tests = subsampling_set(X_test, y_test)

        modeling_set = Vset(name='modeling',
                            modules=[LogisticRegression(max_iter=1000, tol=0.1),
                                     DecisionTreeClassifier()],
                            module_keys=["LR", "DT"])

        modeling_set.fit(X_trains, y_trains)
        preds_test = modeling_set.predict(X_tests)

        # subsampling in (X_trains, y_trains), should not match subsampling in
        # X_tests because they are unrelated
        assert len(modeling_set.out.keys()) == 7
        assert len(preds_test.keys()) == 19

    def test_lazy_eval(self):
        def f(arg_name: str = '', i: int = 0):
            return arg_name, f'f_iter={i}'

        f_modules = [partial(f, i=i) for i in range(3)]
        f_arg = init_args(('f_arg',), names=['f_init'])[0]

        f_set = Vset('f', modules=f_modules)
        f_lazy_set = Vset('f', modules=f_modules, lazy=True)

        f_res = f_set(f_arg)
        f_lazy_res = f_lazy_set(f_arg)

        assert_equal(f_res.keys(), f_lazy_res.keys())

        def g(tup, arg_name: str = '', i: int = 0):
            return tup, arg_name, f'g_iter={i}'

        g_modules = [partial(g, i=i) for i in range(2)]
        g_arg = init_args(('g_arg',), names=['g_init'])[0]

        g_set = Vset('g', modules=g_modules)
        g_lazy_set = Vset('g', modules=g_modules, lazy=True)

        g_res = g_set(f_res, g_arg, n_out=1)
        g_lazy_res = g_lazy_set(f_lazy_res, g_arg, n_out=1)

        assert_equal(g_res.keys(), g_lazy_res.keys())

        def h(tup, arg_name: str = '', i: int = 0):
            return tup, arg_name, f'h_iter={i}'

        h_modules = [partial(h, i=i) for i in range(2)]
        h_arg = init_args(('h_arg',), names=['h_init'])[0]

        h_set = Vset('h', modules=h_modules)

        h_res = h_set(g_res, h_arg, n_out=1)
        h_lazy_res = h_set(g_lazy_res, h_arg, n_out=1)

        # check PREV_KEYs
        assert_equal(h_res[PREV_KEY][0], h_lazy_res[PREV_KEY][0])
        assert_equal(h_res[PREV_KEY][1][0], g_set)
        assert_equal(h_lazy_res[PREV_KEY][1][0], g_lazy_set)
        assert_equal(h_res[PREV_KEY][1][1][0], f_set)
        assert_equal(h_lazy_res[PREV_KEY][1][1][0], f_lazy_set)

        del h_res[PREV_KEY]
        del h_lazy_res[PREV_KEY]

        assert_equal(h_res, h_lazy_res)

    def test_caching(self):
        try:
            np.random.seed(13)
            X, _ = make_classification(n_samples=50, n_features=5)
            X = init_args([X], names=['X'])[0]

            subsampling_funcs = [partial(costly_compute, row_index=np.arange(25))]

            uncached_set = Vset(name='subsampling', modules=subsampling_funcs)
            cached_set = Vset(name='subsampling', modules=subsampling_funcs, cache_dir='./')

            # this always takes about 1 seconds
            begin = time.time()
            uncached_set.fit(X)
            assert time.time() - begin >= 1

            # the first time this runs it takes 1 seconds
            begin = time.time()
            cached_set.fit(X)
            assert time.time() - begin >= 1

            assert_equal(uncached_set.out.keys(), cached_set.out.keys())

            # this should be very fast because it's using the already cached results
            cached_set2 = Vset(name='subsampling', modules=subsampling_funcs, cache_dir='./')
            begin = time.time()
            cached_set2.fit(X)
            assert time.time() - begin < 1
            assert_equal(uncached_set.out.keys(), cached_set2.out.keys())

        finally:
            # clean up
            rmtree('./joblib')

    def test_mlflow_tracking(self, tmp_path):
        try:
            runs_path = os.path.join(tmp_path, 'mlruns')
            np.random.seed(13)
            X, y = make_classification(n_samples=50, n_features=5)
            X_train, X_test, y_train, y_test = init_args(train_test_split(X, y, random_state=42),
                                                         names=['X_train', 'X_test', 'y_train', 'y_test'])
            # fit models
            modeling_set = Vset(name='modeling',
                                modules=[LogisticRegression(C=1, max_iter=1000, tol=0.1)],
                                module_keys=["LR"])

            _ = modeling_set.fit(X_train, y_train)
            preds_test = modeling_set.predict(X_test)
            hard_metrics_set = Vset(name='hard_metrics',
                                    modules=[accuracy_score, balanced_accuracy_score],
                                    module_keys=["Acc", "Bal_Acc"],
                                    tracking_dir=runs_path)
            hard_metrics = hard_metrics_set.evaluate(y_test, preds_test)
            runs_path = os.path.join(runs_path, '1')
            assert os.path.isdir(runs_path)
            assert len(os.listdir(runs_path)) == 2
            runs_path = os.path.join(runs_path, [d for d in os.listdir(runs_path) if d != 'meta.yaml'][0])
            runs_path = os.path.join(runs_path, 'metrics')
            with open(os.path.join(runs_path, 'Acc')) as acc:
                assert len(acc.read().split(" ")) == 3
            with open(os.path.join(runs_path, 'Bal_Acc')) as bal_acc:
                assert len(bal_acc.read().split(" ")) == 3
        finally:
            # clean up
            rmtree(tmp_path)

    def test_async(self):
        def gen_data(n):
            return np.random.randn(n)

        def fun1(a, b=1):
            return a + b

        def fun2(a, b=1):
            return a * b

        data_param_dict = {'n': [1, 2, 3]}
        data_vset = build_vset('data', gen_data, param_dict=data_param_dict, reps=5, lazy=True)

        assert len(data_vset.modules) == 15

        fun_param_dict = {'b': [1, 2, 3]}
        fun1_vset = build_vset('fun1', fun1, param_dict=fun_param_dict, lazy=True)
        fun1_vset_async = build_vset('fun1', fun1, param_dict=fun_param_dict, lazy=True, is_async=True)
        fun2_vset = build_vset('fun2', fun2, param_dict=fun_param_dict)
        fun2_vset_async = build_vset('fun2', fun2, param_dict=fun_param_dict, is_async=True)

        np.random.seed(13)
        ray.init(local_mode=True)

        data = data_vset()

        fun1_res = fun1_vset(data)
        fun1_res_async = fun1_vset_async(data)

        fun2_res = fun2_vset(fun1_res)
        fun2_res_async = fun2_vset_async(fun1_res_async)

        ray.shutdown()

        assert_equal(fun1_res[PREV_KEY][0], fun1_vset)
        assert_equal(fun1_res_async[PREV_KEY][0], fun1_vset_async)
        assert_equal(fun1_res[PREV_KEY][1][0], fun1_res_async[PREV_KEY][1][0])
        assert_equal(fun2_res[PREV_KEY][0], fun2_vset)
        assert_equal(fun2_res_async[PREV_KEY][0], fun2_vset_async)
        assert_equal(fun2_res[PREV_KEY][1][0], fun1_vset)
        assert_equal(fun2_res_async[PREV_KEY][1][0], fun1_vset_async)
        assert_equal(fun2_res[PREV_KEY][1][1][0], fun2_res_async[PREV_KEY][1][1][0])

        del fun2_res[PREV_KEY]
        del fun2_res_async[PREV_KEY]

        assert_equal(fun2_res, fun2_res_async)

    def test_lazy_async(self):
        class learner:
            def fit(self, a):
                self.a = a
                return self
            def transform(self, b):
                return self.a + b
            def predict(self, x):
                return self.a*x
            def predict_proba(self, x):
                y = np.exp(-self.a*x)
                return 1 / (1 + y)

        vset = Vset("learner", [learner()], is_async=True, lazy=True)
        vset.fit(*init_args([.4]))
        data = init_args([np.array([1, 2, 3])])[0]
        transformed = vset.transform(data)
        preds = vset.predict(transformed)
        preds_proba = vset.predict_proba(transformed)

        assert_equal(list(transformed.values())[0](), [1.4, 2.4, 3.4])
        assert_equal(list(preds.values())[0](), np.array([1.4, 2.4, 3.4])*.4)
        assert_equal(list(preds_proba.values())[0](),
                     1 / (1 + np.exp(-np.array([1.4, 2.4, 3.4])*.4)))


def costly_compute(data, row_index=0):
    """Simulate an expensive computation"""
    time.sleep(1)
    return data[row_index,]
