'''Useful functions for converting between different types (dicts, lists, tuples, etc.)
'''
from copy import deepcopy

from vflow.vset import PREV_KEY
from vflow.smart_subkey import SmartSubkey
import pandas as pd
import numpy as np
from pandas import DataFrame


def init_args(args_tuple: tuple, names=None):
    ''' converts tuple of arguments to a list of dicts
    Params
    ------
    names: optional, list-like
        gives names for each of the arguments in the tuple
    '''
    if names is None:
        names = ['start'] * len(args_tuple)
    else:
        assert len(names) == len(args_tuple), 'names should be same length as args_tuple'

    output_dicts = []
    for (i, ele) in enumerate(args_tuple):
        output_dicts.append({
            (SmartSubkey(names[i], 'init'), ): args_tuple[i],
            PREV_KEY: ('init', ),
        })
    return output_dicts


def s(x):
    '''Gets shape of a list/tuple/ndarray
    '''
    if type(x) in [list, tuple]:
        return len(x)
    else:
        return x.shape

def init_step(idx, cols):
    for i in range(idx, len(cols)):
        if cols[i] != 'init':
            return 'init-' + cols[i]

def dict_to_df(d: dict):
    '''Converts a dictionary with tuple keys
    into a pandas DataFrame
    '''
    d_copy = {k:d[k] for k in d if k != PREV_KEY}
    df = pd.Series(d_copy).reset_index()
    if len(d_copy.keys()) > 0:
        cols = [sk.origin for sk in list(d_copy.keys())[0]] + ['out']
        # set each init col to init-{next_module_set}
        cols = [c if c != 'init' else init_step(idx, cols) for idx, c in enumerate(cols) ]
        df.set_axis(cols, axis=1, inplace=True)
    return df

def compute_interval(df: DataFrame, d_label, wrt_label, accum: list=['std']):
    '''Compute an interval (std. dev) of d_label column with 
    respect to pertubations in the wrt_label column
    TODO: Add fn param to set accum type
    '''
    return df[[wrt_label, d_label]].groupby(wrt_label).agg(accum)

def predict_interval(preds: dict, y_real: dict):
    preds = {k: v for k, v in preds.items() if k != PREV_KEY}
    y_real = {k: v for k, v in y_real.items() if k != PREV_KEY}
    preds_arr = np.array([(l - np.mean(l)) / np.std(l) for l in list(preds.values())])
    uncertainty = np.std(preds_arr, axis=0)
    binned_acc = 1 - np.sum(np.abs(preds_arr - list(y_real.values())[0]), axis=0) / preds_arr.shape[0]
    return uncertainty, binned_acc

def to_tuple(lists: list):
    '''Convert from lists to unpacked  tuple
    Ex. [[x1, y1], [x2, y2], [x3, y3]] -> ([x1, x2, x3], [y1, y2, y3])
    Ex. [[x1, y1]] -> ([x1], [y1])
    Ex. [m1, m2, m3] -> [m1, m2, m3]
    Allows us to write X, y = ([x1, x2, x3], [y1, y2, y3])
    '''
    n_mods = len(lists)
    if n_mods <= 1:
        return lists
    if not type(lists[0]) == list:
        return lists
    n_tup = len(lists[0])
    tup = [[] for _ in range(n_tup)]
    for i in range(n_mods):
        for j in range(n_tup):
            tup[j].append(lists[i][j])
    return tuple(tup)


def to_list(tup: tuple):
    '''Convert from tuple to packed list
    Ex. ([x1, x2, x3], [y1, y2, y3]) -> [[x1, y1], [x2, y2], [x3, y3]]
    Ex. ([x1], [y1]) -> [[x1, y1]]
    Ex. ([x1, x2, x3]) -> [[x1], [x2], [x3]]
    Ex. (x1) -> [[x1]]
    Ex. (x1, y1) -> [[x1, y1]]
    Ex. (x1, x2, x3, y1, y2, y3) -> [[x1, y1], [x2, y2], [x3, y3]]
    Ex. (x1, x2, x3, y1, y2) -> Error
    Allows us to call function with arguments in a loop
    '''
    n_tup = len(tup)
    if n_tup == 0:
        return []
    elif not isinstance(tup[0], list):
        # the first element is data
        if n_tup == 1:
            return list(tup)
        if n_tup % 2 != 0:
            raise ValueError('Don\'t know how to handle uneven number of args '
                             'without a list. Please wrap your args in a list.')
        # assume first half of args is input and second half is outcome
        return [list(el) for el in zip(tup[:(n_tup // 2)], tup[(n_tup // 2):])]
    elif n_tup == 1:
        return [[x] for x in tup[0]]
    n_mods = len(tup[0])
    lists_packed = [[] for _ in range(n_mods)]
    for i in range(n_mods):
        for j in range(n_tup):
            lists_packed[i].append(tup[j][i])
    return lists_packed


def sep_dicts(d: dict, n_out: int = 1):
    '''converts dictionary with value being saved as an iterable into multiple dictionaries
    Assumes every value has same length n_out

    Params
    ------
    d: {k1: (x1, y1), k2: (x2, y2), ...,  '__prev__': p}
    n_out: the number of dictionaries to separate d into

    Returns
    -------
    sep_dicts: [{k1: x1, k2: x2, ..., '__prev__': p}, {k1: y1, k2: y2, '__prev__': p}]
    '''
    # empty dict -- return empty dict
    if n_out == 1:
        return d
    n_dicts = len(d)
    if n_dicts == 0:
        return [{}]
    else:
        # try separating dict into multiple dicts
        val_list_len = len(tuple(d.values())[0])  # first item in list
        sep_dicts = [dict() for x in range(n_out)]
        for key, value in d.items():
            if key != PREV_KEY:
                for i in range(n_out):
                    # assumes the correct sub-key for item i is in the i-th position
                    new_key = (key[i],) + key[n_out:]
                    sep_dicts[i][new_key] = value[i]

        # add back prev
        prev = d[PREV_KEY]
        for i in range(n_out):
            sep_dicts[i][PREV_KEY] = prev
        return sep_dicts


def combine_keys(left_key, right_key):
    if len(left_key) < len(right_key):
        match_key = left_key
        compare_key = right_key
    else:
        match_key = right_key
        compare_key = left_key
    match_smartkeys = [subkey for subkey in match_key if subkey.is_matching()]
    if len(match_smartkeys) > 0:
        matched_subkeys = []
        for subkey in match_smartkeys:
            for c_subkey in compare_key:
                if subkey.is_matching():
                    if subkey.origin == c_subkey.origin:
                        if subkey.subkey == c_subkey.subkey:
                            matched_subkeys.append(subkey)
                            break
                        else:
                            # subkeys with same origin but different values are rejected
                            return ()
        if len(matched_subkeys) > 0:
            # always filter on right key
            filtered_key = tuple([subkey for subkey in right_key if subkey not in matched_subkeys])
            combined_key = left_key + filtered_key
            return combined_key
        else:
            return left_key + right_key
    else:
        return left_key + right_key


def combine_dicts(*args: dict, base_case=True):
    '''Combines any number of dictionaries into a single dictionary. Dictionaries
    are combined left to right, matching on the subkeys of the arg that has
    fewer matching requirements.
    '''
    n_args = len(args)
    combined_dict = {}
    if n_args == 0:
        return combined_dict
    elif n_args == 1:
        for k in args[0]:
            # wrap the dict values in tuples; this is helpful so that when we
            # pass the values to a module fun in we just can use * expansion
            if k != PREV_KEY:
                combined_dict[k] = (args[0][k],)
            else:
                combined_dict[k] = args[0][k]
        return combined_dict
    elif n_args == 2:
        for k0 in args[0]:
            for k1 in args[1]:

                if k0 == PREV_KEY or k1 == PREV_KEY:
                    continue

                combined_key = combine_keys(k0, k1)

                if len(combined_key) > 0:
                    if base_case:
                        combined_dict[combined_key] = (args[0][k0], args[1][k1])
                    else:
                        combined_dict[combined_key] = args[0][k0] + (args[1][k1],)

        prev_tup = ()
        for i in range(2):
            if PREV_KEY in args[i]:
                prev_tup += args[i][PREV_KEY]
        combined_dict[PREV_KEY] = prev_tup
        return combined_dict
    else:
        # combine the first two dicts and call recursively with remaining args
        return combine_dicts(combine_dicts(args[0], args[1]), *args[2:], base_case=False)


def apply_modules(modules: dict, data_dict: dict):
    out_dict = {}
    for mod_k in modules:
        for data_k in data_dict:
            if mod_k == PREV_KEY or data_k == PREV_KEY:
                continue

            combined_key = combine_keys(data_k, mod_k)

            if len(combined_key) > 0:
                out_dict[combined_key] = deepcopy(modules[mod_k])(*data_dict[data_k])

    return out_dict
