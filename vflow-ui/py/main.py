import sys
import json
from vflow import init_args, Vset, build_vset
import pandas as pd

arg = sys.argv[1]

payload = json.loads(arg)
print(f"Executing {payload['cmd']} command", flush=True)
print(f"path to the data \n {payload['data_path']}", flush=True)

if payload.init:
    # handle payload
    X = pd.read_csv(payload.train_data_path)
    y = pd.read_csv(payload.test_data_path)
    X_train, X_test, y_train, y_test = init_args(
        sklearn.model_selection.train_test_split(X, y),
        names=['X_train', 'X_test', 'y_train', 'y_test']  # optionally name the args
    )

    if payload.subsample:
        # subsample data
        subsampling_funcs = [
            sklearn.utils.resample for _ in range(3)
        ]
        subsampling_set = build_vset(name='subsampling',
                               funcs=subsampling_funcs,
                               output_matching=True)
        X_trains, y_trains = subsampling_set(X_train, y_train)

if payload.train:
    model_set = [get_imported_module(m)() for m in payload.model_names]
    modeling_set = build_vset(name = 'modeling', funcs=model_set)
    modeling_set.fit(X_trains, y_trains)


def get_imported_module(class_name):
    """ return class module """

sys.stdout.flush()
