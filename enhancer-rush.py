# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pcsp
from pcsp import PCSPipeline, ModuleSet, Module, init_args
from pcsp.pipeline import build_graph
from functools import partial
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import sklearn.utils
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from irf import irf_utils, irf_jupyter_utils
from irf.irf_utils import run_iRF
from irf.ensemble import RandomForestClassifierWithWeights
from sklearn.inspection import permutation_importance
import sys

# %%
# load data
X_train = np.asarray(pd.read_csv("./data/01_X_train.csv",error_bad_lines=False).iloc[:,1:])
X_test = np.asarray(pd.read_csv("./data/02_X_test.csv",error_bad_lines=False).iloc[:,1:])
y_train = np.asarray(pd.read_csv("./data/03_y_train.csv",error_bad_lines=False).iloc[:,1])
y_test = np.asarray(pd.read_csv("./data/04_y_test.csv",error_bad_lines=False).iloc[:,1])

rf = RandomForestClassifier(n_estimators=100, max_depth=5).fit(X_train, y_train)
# initialize data
np.random.seed(14)
X_train, X_test, y_train, y_test = init_args((X_train, X_test, y_train, y_test), names=['X_train', 'X_test', 'y_train', 'y_test'])

# subsample
subsampling_fns = [partial(sklearn.utils.resample, n_samples=1000, random_state=i) for i in range(3)]

subsampling_set = ModuleSet(name='subsampling', modules=subsampling_fns)
X_trains, y_trains = subsampling_set(X_train, y_train)

modeling_set = ModuleSet(name='modeling', modules=[RandomForestClassifier(n_estimators=50, max_depth=5)], module_keys=["RF"])

# model
modeling_set.fit(X_trains, y_trains)
preds = modeling_set.predict(X_test)

# hard metrics
hard_metrics_set = ModuleSet(name='hard_metrics', modules=[accuracy_score, balanced_accuracy_score], module_keys=["Acc", "Bal_Acc"])
hard_metrics = hard_metrics_set.evaluate(preds, y_test)

# permutation importance
def perm_importance(data, estimator):
    X, y = data
    return permutation_importance(estimator, X, y)

feature_importance_set = ModuleSet(name='feature_importance', modules=[perm_importance])
importances = feature_importance_set.evaluate(modeling_set.out, X_test, y_test)

G = build_graph(hard_metrics, draw=True)
plt.show()
importances


# %%
print({k:np.argsort(v.importances_mean)[-5:][::-1] for (k, v) in importances.items() if type(v) != pcsp.module_set.ModuleSet})

# %%
print(modeling_set.out.keys())
print(X_test.keys())
print(y_test.keys())


# %%



