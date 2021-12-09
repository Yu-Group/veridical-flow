"""Useful functions for converting between different types (dicts, lists, tuples, etc.)
"""
from copy import deepcopy
from typing import Union
from uuid import uuid4

import pandas as pd
from pandas import DataFrame

from vflow.subkey import Subkey
from vflow.vfunc import VfuncPromise
from vflow.vset import PREV_KEY


def init_args(args_tuple: Union[tuple, list], names=None):
    """ converts tuple of arguments to a list of dicts
    Params
    ------
    names: optional, list-like
        gives names for each of the arguments in the tuple
    """
    if names is None:
        names = ['start'] * len(args_tuple)
    else:
        assert len(names) == len(args_tuple), 'names should be same length as args_tuple'

    output_dicts = []
    for (i, ele) in enumerate(args_tuple):
        output_dicts.append({
            (Subkey(names[i], 'init'),): args_tuple[i],
            PREV_KEY: ('init',),
        })
    return output_dicts


def s(x):
    """Gets shape of a list/tuple/ndarray
    """
    if type(x) in [list, tuple]:
        return len(x)
    else:
        return x.shape


def init_step(idx, cols):
    for i in range(idx, len(cols)):
        if cols[i] != 'init':
            return 'init-' + cols[i]


def dict_to_df(d: dict, param_key=None):
    """Converts a dictionary with tuple keys
    into a pandas DataFrame, optionally seperating
    parameters in param_key if not None
    """
    d_copy = {tuple([sk.value for sk in k]): d[k] for k in d if k != PREV_KEY}
    df = pd.Series(d_copy).reset_index()
    if len(d_copy.keys()) > 0:
        key_list = list(d.keys())
        subkey_list = key_list[0] if key_list[0] != PREV_KEY else key_list[1]
        cols = [sk.origin for sk in subkey_list] + ['out']
        # set each init col to init-{next_module_set}
        cols = [c if c != 'init' else init_step(idx, cols) for idx, c in enumerate(cols)]
        df.set_axis(cols, axis=1, inplace=True)
        if param_key:
            param_keys = df[param_key].tolist()
            if param_key == 'out' and hasattr(param_keys[0], '__iter__'):
                param_df = pd.DataFrame(param_keys)
                param_df.columns = [f'{param_key}-{col}' for col in param_df.columns]
                df = df.join(param_df)
            else:
                param_loc = df.columns.get_loc(param_key)
                param_key_cols = [f"{p.split('=')[0]}-{param_key}" for p in param_keys[0]]
                param_keys = [[s.split('=')[1] for s in t] for t in param_keys]
                df = df.join(pd.DataFrame(param_keys)).drop(columns=param_key)
                new_cols = df.columns[:len(cols)-1].tolist() + param_key_cols
                df.set_axis(new_cols, axis=1, inplace=True)
                new_idx = list(range(len(new_cols)))
                new_idx = new_idx[:param_loc] + new_idx[len(cols)-1:] + new_idx[param_loc:len(cols)-1]
                df = df.iloc[:, new_idx]
    return df


def compute_interval(df: DataFrame, d_label, wrt_label, accum=None):
    """Compute an interval (std. dev) of d_label column with
    respect to pertubations in the wrt_label column
    """
    if accum is None:
        accum = ['std']
    df = df.astype({wrt_label: str})
    return df[[wrt_label, d_label]].groupby(wrt_label).agg(accum)


def perturbation_stats(df: DataFrame, *groups: str, wrt_col: str = 'out',
                       func=None):
    """Compute statistics for wrt_col in df, conditional on groups
    """
    if func is None:
        func = ['count', 'mean', 'std']
    groups = list(groups)
    df = df.groupby(groups).agg(func)[wrt_col]
    df.reset_index(inplace=True)
    return df.sort_values(groups[0])


def to_tuple(lists: list):
    """Convert from lists to unpacked  tuple
    Ex. [[x1, y1], [x2, y2], [x3, y3]] -> ([x1, x2, x3], [y1, y2, y3])
    Ex. [[x1, y1]] -> ([x1], [y1])
    Ex. [m1, m2, m3] -> [m1, m2, m3]
    Allows us to write X, y = ([x1, x2, x3], [y1, y2, y3])
    """
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
    """Convert from tuple to packed list
    Ex. ([x1, x2, x3], [y1, y2, y3]) -> [[x1, y1], [x2, y2], [x3, y3]]
    Ex. ([x1], [y1]) -> [[x1, y1]]
    Ex. ([x1, x2, x3]) -> [[x1], [x2], [x3]]
    Ex. (x1) -> [[x1]]
    Ex. (x1, y1) -> [[x1, y1]]
    Ex. (x1, x2, x3, y1, y2, y3) -> [[x1, y1], [x2, y2], [x3, y3]]
    Ex. (x1, x2, x3, y1, y2) -> Error
    Allows us to call function with arguments in a loop
    """
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


def sep_dicts(d: dict, n_out: int = 1, keys=None):
    """converts dictionary with value being saved as an iterable into multiple dictionaries
    Assumes every value has same length n_out

    Params
    ------
    d: {k1: (x1, y1), k2: (x2, y2), ...,  '__prev__': p}
    n_out: the number of dictionaries to separate d into

    Returns
    -------
    sep_dicts: [{k1: x1, k2: x2, ..., '__prev__': p}, {k1: y1, k2: y2, '__prev__': p}]
    """
    if keys is None:
        keys = []
    if len(keys) > 0 and len(keys) != n_out:
        raise ValueError(f'keys should be empty or have length n_out={n_out}')
    # empty dict -- return empty dict
    if n_out <= 1:
        return d
    else:
        # try separating dict into multiple dicts
        sep_dicts_id = str(uuid4())  # w/ high prob, uuid4 is unique
        sep_dicts_list = [dict() for _ in range(n_out)]
        for key, value in d.items():
            if key != PREV_KEY:
                for i in range(n_out):
                    # assumes the correct sub-key for item i is in the i-th position
                    if len(keys) == 0:
                        new_key = (key[i],) + key[n_out:]
                    else:
                        new_sub = Subkey(value=keys[i], origin=key[-1].origin + '-' + str(i))
                        new_key = (new_sub,) + key
                    new_key[-1]._sep_dicts_id = sep_dicts_id
                    if isinstance(value, VfuncPromise):
                        # return a promise to get the value at index i of the
                        # original promise
                        value_i = VfuncPromise(lambda v, x: v[x], value, i)
                    else:
                        value_i = value[i]
                    sep_dicts_list[i][new_key] = value_i

        return sep_dicts_list


def combine_keys(left_key, right_key):
    if len(left_key) < len(right_key):
        match_key = left_key
        compare_key = right_key
    else:
        match_key = right_key
        compare_key = left_key
    match_subkeys = [subkey for subkey in match_key if subkey.is_matching()]
    if len(match_subkeys) > 0:
        matched_subkeys = []
        for subkey in match_subkeys:
            for c_subkey in compare_key:
                if subkey.matches(c_subkey):
                    matched_subkeys.append(subkey)
                    break
                elif subkey.mismatches(c_subkey):
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
    """Combines any number of dictionaries into a single dictionary. Dictionaries
    are combined left to right, matching on the subkeys of the arg that has
    fewer matching requirements.
    """
    n_args = len(args)
    combined_dict = {}
    if n_args == 0:
        return combined_dict
    elif n_args == 1:
        for k in args[0]:
            # wrap the dict values in tuples; this is helpful so that when we
            # pass the values to a module fun in we can just use * expansion
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

        return combined_dict
    else:
        # combine the first two dicts and call recursively with remaining args
        return combine_dicts(combine_dicts(args[0], args[1]), *args[2:], base_case=False)


def apply_modules(modules: dict, data_dict: dict, lazy: bool = False):
    out_dict = {}
    for mod_k in modules:
        if len(data_dict) == 0:
            func = deepcopy(modules[mod_k])
            if lazy:
                out_dict[mod_k] = VfuncPromise(func)
            else:
                out_dict[mod_k] = func()
        for data_k in data_dict:
            if mod_k == PREV_KEY or data_k == PREV_KEY:
                continue

            combined_key = combine_keys(data_k, mod_k)

            if len(combined_key) > 0:
                func = deepcopy(modules[mod_k])
                if lazy:
                    # return a promise
                    out_dict[combined_key] = VfuncPromise(func, *data_dict[data_k])
                else:
                    data_list = list(data_dict[data_k])
                    for i, data in enumerate(data_list):
                        if isinstance(data, VfuncPromise):
                            data_list[i] = data()
                    out_dict[combined_key] = func(*data_list)

    return out_dict
