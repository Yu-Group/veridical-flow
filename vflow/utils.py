"""Useful functions for converting between different types (dicts, lists, tuples, etc.)
"""
from copy import deepcopy
from typing import Union
from uuid import uuid4

import numpy as np
import pandas as pd

import ray
from ray.remote_function import RemoteFunction as RayRemoteFun

from vflow.subkey import Subkey
from vflow.vfunc import VfuncPromise


PREV_KEY = '__prev__'

def s(x):
    """Gets shape of a list/tuple/ndarray
    """
    if type(x) in [list, tuple]:
        return len(x)
    return x.shape


def init_step(idx, cols):
    """Helper function to find init suffix
    in a column

    Parameters
    ----------
    idx: int
        Index of 'init' column in cols.
    cols: list[str]
        List of column names.
    """
    for i in range(idx, len(cols)):
        if cols[i] != 'init':
            return 'init-' + cols[i]
    return None


def base_dict(d: dict):
    """Remove PREV_KEY from dict d if present
    """
    return {k:v for k,v in d.items() if k != PREV_KEY}


def dict_data(d: dict):
    """Returns a list containing all data in dict d
    """
    return list(base_dict(d).values())


def dict_keys(d: dict):
    """Returns a list containing all keys in dict d
    """
    return list(base_dict(d).keys())


def to_tuple(lists: list):
    """Convert from lists to unpacked tuple

    Allows us to write `X, y = to_tuple([[x1, y1], [x2, y2], [x3, y3]])`

    Parameters
    ----------
    lists: list
        list of objects to convert to unpacked tuple

    Examples
    --------
    >>> to_tuple([[x1, y1], [x2, y2], [x3, y3]])
    ([x1, x2, x3], [y1, y2, y3])
    >>> to_tuple([[x1, y1]])
    ([x1], [y1])
    >>> to_tuple([m1, m2, m3])
    [m1, m2, m3]
    """
    n_mods = len(lists)
    if n_mods <= 1:
        return lists
    if not isinstance(lists[0], list):
        return lists
    n_tup = len(lists[0])
    tup = [[] for _ in range(n_tup)]
    for i in range(n_mods):
        for j in range(n_tup):
            tup[j].append(lists[i][j])
    return tuple(tup)


def to_list(tup: tuple):
    """Convert from tuple to packed list

    Allows us to call function with arguments in a loop

    Parameters
    ----------
    tup: tuple
        tuple of objects to convert to packed list

    Raises
    ------
    ValueError
        If passed uneven number of arguments without a list. Please wrap your args in a list.

    Examples
    --------
    >>> to_list(([x1, x2, x3], [y1, y2, y3]))
    [[x1, y1], [x2, y2], [x3, y3]]
    >>> to_list(([x1], [y1]))
    [[x1, y1]]
    >>> to_list(([x1, x2, x3], ))
    [[x1], [x2], [x3]]
    >>> to_list((x1, ))
    [[x1]]
    >>> to_list((x1, y1))
    [[x1, y1]]
    >>> to_list((x1, x2, x3, y1, y2, y3))
    [[x1, y1], [x2, y2], [x3, y3]]
    """
    n_tup = len(tup)
    if n_tup == 0:
        return []
    if not isinstance(tup[0], list):
        # the first element is data
        if n_tup == 1:
            return [list(tup)]
        if n_tup % 2 != 0:
            raise ValueError('Don\'t know how to handle uneven number of args '
                             'without a list. Please wrap your args in a list.')
        # assume first half of args is input and second half is outcome
        return [list(el) for el in zip(tup[:(n_tup // 2)], tup[(n_tup // 2):])]
    if n_tup == 1:
        return [[x] for x in tup[0]]
    n_mods = len(tup[0])
    lists_packed = [[] for _ in range(n_mods)]
    for i in range(n_mods):
        for j in range(n_tup):
            lists_packed[i].append(tup[j][i])
    return lists_packed


def sep_dicts(d: dict, n_out: int = 1, keys=None):
    """Converts dictionary with value being saved as an iterable into multiple dictionaries

    Assumes every value has same length n_out

    Parameters
    ----------
    d: dict
        Dictionary with iterable values to be converted.
    n_out: int, default 1
        The number of dictionaries to separate d into.
    keys: list-like, default None
        Optional list of keys to use in output dicts.

    Returns
    -------
    sep_dicts_list: list
        List of seperated dictionaries.

    Examples
    --------
    >>> sep_dicts({k1: (x1, y1), k2: (x2, y2), ...,  '__prev__': p})
    [{k1: x1, k2: x2, ..., '__prev__': p}, {k1: y1, k2: y2, ..., '__prev__': p}]
    """
    if keys is None:
        keys = []
    if len(keys) > 0 and len(keys) != n_out:
        raise ValueError(f'keys should be empty or have length n_out={n_out}')
    # empty dict -- return empty dict
    if n_out <= 1:
        return d
    # try separating dict into multiple dicts
    sep_dicts_id = str(uuid4())  # w/ high prob, uuid4 is unique
    sep_dicts_list = [{} for _ in range(n_out)]
    for key, value in d.items():
        if key != PREV_KEY:
            for i in range(n_out):
                # assumes the correct sub-key for item i is in the i-th position
                if len(keys) == 0:
                    new_key = (key[i],) + key[n_out:]
                else:
                    new_sub = Subkey(value=keys[i], origin=key[-1].origin + '-' + str(i))
                    new_key = (new_sub,) + key
                new_key[-1].sep_dicts_id = sep_dicts_id
                if isinstance(value, VfuncPromise):
                    # return a promise to get the value at index i of the
                    # original promise
                    value_i = VfuncPromise(lambda v, x: v[x], value, i)
                else:
                    value_i = value[i]
                sep_dicts_list[i][new_key] = value_i

    return sep_dicts_list

def dict_to_df(d: dict, param_key=None):
    """Converts a dictionary with tuple keys
    into a pandas DataFrame, optionally seperating
    parameters in `param_key` if not None

    Parameters
    ----------
    d: dict
        Output dictionary with tuple keys from a Vset.
    param_key: str (optional), default None
        Name of parameter to seperate into multiple columns.

    Returns
    -------
    df: pandas.DataFrame
        A DataFrame with `d` tuple keys seperated into columns.
    """
    d_copy = {tuple(sk.value for sk in k): d[k] for k in d if k != PREV_KEY}
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


def perturbation_stats(data: Union[pd.DataFrame, dict], *group_by: str, wrt: str = 'out',
                       func=None, prefix: str = None, split: bool = False):
    """Compute statistics for `wrt` in `data`, conditional on `group_by`

    Parameters
    ----------
    data: Union[pandas.DataFrame, dict]
        DataFrame, as from calling `dict_to_df` on an output dict from a Vset,
        or the output dict itself.
    *group_by: str
        Vset names in `data` to group on. If none provided, treats everything as one big
        group.
    wrt: str (optional)
        Column name in `data` or `dict_to_df(data)` on which to compute statistics.
        Defaults to `'out'`, the values of the original Vset output dict.
    func: function, str, list or dict (optional), default None
        A list of functions or function names to use for computing
        statistics, analogous to the parameter of the same name in
        pandas.core.groupby.DataFrameGroupBy.aggregate. If `None`, defaults to
        `['count', 'mean', 'std']`.
    prefix: str (optional), default None
        A string to prefix to new columns in output DataFrame. If `None`,
        uses the value of `wrt`.
    split: bool (optional), default False
        If `True` and `wrt` in `data` has `list` or `numpy.ndarray` entries, will
        attempt to split the entries into multiple columns for the output.

    Returns
    -------
    df: pandas.DataFrame
        A DataFrame with summary statistics on `wrt`.
    """
    if func is None:
        func = ['count', 'mean', 'std']
    if prefix is None:
        prefix = wrt
    if isinstance(data, dict):
        df = dict_to_df(data)
    else:
        df = data
    group_by = list(group_by)
    if len(group_by) > 0:
        gb = df.groupby(group_by)[wrt]
    else:
        gb = df.groupby(lambda x: True)[wrt]
    if (isinstance(func, list) and 'mean' in func or 'std' in func) and \
       (type(df[wrt].iloc[0]) in [list, np.ndarray]):
        wrt_arrays = [np.stack(d.tolist()) for d in (gb.get_group(grp) for grp in gb.groups)]
        n_cols = wrt_arrays[0].shape[1]
        df_out = pd.DataFrame(gb.agg('count'))
        df_out.columns = [f'{prefix}-count']
        if 'mean' in func:
            if split:
                col_means = [arr.mean(axis=0) for arr in wrt_arrays]
                wrt_means = pd.DataFrame(col_means,
                                         columns=[f'{prefix}{i}-mean' for i in range(n_cols)],
                                         index=gb.groups.keys())
            else:
                col_means = [{f'{prefix}-mean': arr.mean(axis=0)} for arr in wrt_arrays]
                wrt_means = pd.DataFrame(col_means, index=gb.groups.keys())
            wrt_means.index.names = df_out.index.names
            df_out = df_out.join(wrt_means)
        if 'std' in func:
            if split:
                col_stds = [arr.std(axis=0, ddof=1) for arr in wrt_arrays]
                wrt_stds = pd.DataFrame(col_stds,
                                        columns=[f'{prefix}{i}-std' for i in range(n_cols)],
                                        index=gb.groups.keys())
            else:
                col_stds = [{f'{prefix}-std': arr.std(axis=0, ddof=1)} for arr in wrt_arrays]
                wrt_stds = pd.DataFrame(col_stds, index=gb.groups.keys())
            wrt_stds.index.names = df_out.index.names
            df_out = df_out.join(wrt_stds)
        if 'count' not in func:
            df_out = df_out.drop(f'{prefix}-count')
    else:
        df_out = gb.agg(func)
    df_out = df_out.reindex(sorted(df_out.columns), axis=1)
    df_out.reset_index(inplace=True)
    if len(group_by) > 0:
        return df_out.sort_values(group_by[0])
    return df_out


def combine_keys(left_key, right_key):
    """Combines `left_key` and `right_key`, attempting to match on any `vflow.subkey.Subkey.is_matching`

    Returns an empty key on failed matches (`Subkey` with same origin but different values).
    Always filters on `right_key` and returns `combined_key` with `left_key` prefix.

    Parameters
    ----------
    left_key: tuple
        Left tuple key to combine.
    right_key: tuple
        Right tuple key to combine.

    Returns
    -------
    combined_key: tuple
        Combined tuple key filtered according to `vflow.subkey.Subkey.matches` rules,
        which is empty according to `vflow.subkey.Subkey.mismatches` rule.
    """
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
                if subkey.mismatches(c_subkey):
                    # subkeys with same origin but different values are rejected
                    return ()
        if len(matched_subkeys) > 0:
            # always filter on right key
            filtered_key = tuple(subkey for subkey in right_key if subkey not in matched_subkeys)
            combined_key = left_key + filtered_key
            return combined_key
        return left_key + right_key
    return left_key + right_key


def combine_dicts(*args: dict, base_case=True):
    """Combines any number of dictionaries into a single dictionary. Dictionaries
    are combined left to right matching all keys according to `combine_keys`

    Parameters
    ----------
    *args: dict
        Dictionaries to recursively combine left to right.

    Returns
    -------
    combined_dict: dict
        Combined dictionary.
    """
    n_args = len(args)
    combined_dict = {}
    if n_args == 0:
        return combined_dict
    if n_args == 1:
        for k in args[0]:
            # wrap the dict values in tuples; this is helpful so that when we
            # pass the values to a module fun in we can just use * expansion
            if k != PREV_KEY:
                combined_dict[k] = (args[0][k],)
            else:
                combined_dict[k] = args[0][k]
        return combined_dict
    if n_args == 2:
        for k0 in args[0]:
            for k1 in args[1]:

                if PREV_KEY in (k0, k1):
                    continue

                combined_key = combine_keys(k0, k1)

                if len(combined_key) > 0:
                    if base_case:
                        combined_dict[combined_key] = (args[0][k0], args[1][k1])
                    else:
                        combined_dict[combined_key] = args[0][k0] + (args[1][k1],)

        return combined_dict
    # combine the first two dicts and call recursively with remaining args
    return combine_dicts(combine_dicts(args[0], args[1]), *args[2:], base_case=False)


def apply_modules(modules: dict, data_dict: dict, lazy: bool=False):
    """Apply a dictionary of functions `modules` to each item of `data_dict`,
    optionally returning a dictionary of `vflow.vfunc.VfuncPromise` objects if `lazy` is True

    Output keys are determined by applying `combine_keys` to each pair of items from
    `modules` and `data_dict`. This function is used by all Vsets to apply functions.

    Parameters
    ----------
    modules: dict
        Dictionary of functions to apply to `data_dict`.
    data_dict: dict
        Dictionary of parameters to call each function in `modules`.
    lazy: bool (option), default False
        If True, `modules` are applied lazily, returning `vflow.vfunc.VfuncPromise`
        objects,

    Returns
    -------
    out_dict: dict
        Output dictionary of applying `modules` to `data_dict`.
    """
    out_dict = {}
    for mod_k in modules:
        if len(data_dict) == 0:
            func = deepcopy(modules[mod_k])
            if lazy:
                out_dict[mod_k] = VfuncPromise(func)
            else:
                out_dict[mod_k] = func()
        for data_k in data_dict:
            if PREV_KEY in (mod_k, data_k):
                continue

            combined_key = combine_keys(data_k, mod_k)

            if not len(combined_key) > 0:
                continue

            func = deepcopy(modules[mod_k])
            if lazy:
                # return a promise
                out_dict[combined_key] = VfuncPromise(func, *data_dict[data_k])
            else:
                data_list = list(data_dict[data_k])
                for i, data in enumerate(data_list):
                    if isinstance(data, VfuncPromise):
                        data_list[i] = data()
                    if isinstance(func, RayRemoteFun) and not isinstance(data_list[i], ray.ObjectRef):
                        # send data to Ray's remote object store
                        data_list[i] = ray.put(data_list[i])
                    elif isinstance(data_list[i], ray.ObjectRef):
                        # this is not a remote function so get the data
                        data_list[i] = ray.get(data_list[i])
                out_dict[combined_key] = func(*data_list)

    return out_dict
