import pcsp
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample
from functools import partial
from pcsp import PCSPipeline, ModuleSet, Module # must install pcsp first (pip install pcsp)
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

class TestBasic():
    def setup(self):
        pass

    def test_subsampling_fitting_metrics_pipeline(self):
        np.random.seed(13)
        # subsample data
        X, y = make_classification(n_samples=50, n_features=5)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
        subsampling_funcs = [partial(resample,
                                    n_samples=int(X_train.shape[0]*0.3),
                                    random_state=i)
                             for i in range(3)]
        subsampling_set = ModuleSet(name='subsampling',
                                    modules=subsampling_funcs)
        
        # todo: shouldn't have to pass list for first args
        X_all, y_all = subsampling_set([X_train], [y_train]) # subsampling_set([X_train, X_train], [y_train, y_train]) # artificially make it seem like there are multiple dsets (data_0 and data_1)

        # fit models
        modeling_set = ModuleSet(name='modeling',
                                 modules=[LogisticRegression(max_iter=1000, tol=0.1),
                                          DecisionTreeClassifier()],
                                 module_keys=["LR", "DT"])
        models = modeling_set.fit(X_all, y_all)  # ModuleSet needs to store something for this call to work (makes models kind of useless)

        # get predictions
        X_all["test"] = X_test
        y_all["test"] = y_test
        preds_all = modeling_set.predict(X_all)

        # get metrics
        hard_metrics_set = ModuleSet(name='hard_metrics',
                                     modules=[accuracy_score, balanced_accuracy_score],
                                     module_keys=["Acc", "Bal_Acc"])
        hard_metrics = hard_metrics_set.evaluate(y_all, preds_all)
        
        # asserts
        k1 = (('data_0', 'subsampling_0', 'LR'), ('data_0', 'subsampling_0'), 'Acc')
        assert k1 in hard_metrics, 'hard metrics should have ' + str(k1) + ' as key'
        assert hard_metrics[k1] > 0.9 # 0.9090909090909091
        assert '__prev__' in hard_metrics
        assert len(hard_metrics.keys()) == 49
