import sys
import json
import pickle
from sklearn.model_selection import train_test_split
from vflow import init_args, build_vset
import pandas as pd
from .helpers import *

# get input
full_json_input = sys.argv[1] # in json format
full_input = json.loads(full_json_input) # [{'cmd': 'init', ...}, ..., {'cmd': 'smth', ...}]

# print(f"Executing {full_json_input['cmd']} command", flush=True)    # log
# print(f"path to the data \n {full_json_input['data_path']}", flush=True) # log

pickled_inputs = [] # [(X_train, X_test, y_train, y_test), pickle_vset1, pickle_vset2, ...] 
for op in full_input:
    i = 0
    if op['cmd'] == 'init':
        # init procedure
        X = pd.read_csv(op['X_data_path'])   # specify type of X input (columns = features) 
        y = pd.read_csv(op['y_data_path'])   # specify type of y input (should be 1 column)
        X_train, X_test, y_train, y_test = init_args(train_test_split(X, y, random_state=123),
                                             names=['X_train', 'X_test', 'y_train', 'y_test'])
        pickled_inputs.append((X_train, X_test, y_train, y_test))
    else:
        # gather args for build_vset
        func = get_imported_module(op['class_name'])
        curr_vset = build_vset(name = op['cmd'], func = func, 
                                param_dict = None, reps = op['reps'],
                                is_async = False, output_matching = True,
                                lazy = True, cache_dir = None,
                                tracking_dir = None)
        
        # call vset here and store the result
        if op['cmd'] == 'Preprocess':
            # load prev result
            X_train = pickled_inputs[0][0]
            y_train = pickled_inputs[0][2]
            # execute
            curr_vset_result = curr_vset(X_train, y_train)
        elif op['cmd'] == 'Model':
            # load prev result
            X_trains = pickled_inputs[1]
            y_trains = pickled_inputs[1]
            # execute 
            curr_vset.fit(X_trains, y_trains)
            curr_vset_result = curr_vset.predict(X_test)
        elif op['cmd'] == 'Metrics':
            # load prev result
            preds_test = pickled_inputs[2]
            # execute 
            curr_vset_result = curr_vset.evaluate(preds_test, y_test)

        # store curr result
        file = open(f"pickle/pickle{i}")
        pickled_inputs.append(pickle.dump(curr_vset_result, file))
    
    i += 1
        
# compile outputs of all the vsets 


sys.stdout.flush()
