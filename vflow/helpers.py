"""User-facing helper functions included at import vflow
"""
from functools import partial
from itertools import product
from typing import Union

import pandas as pd
import numpy as np

from vflow.utils import init_step, dict_keys, dict_data
from vflow.vfunc import Vfunc
from vflow.vset import Vset, Subkey, PREV_KEY, FILTER_PREV_KEY

def init_args(args_tuple: Union[tuple, list], names=None):
    """Converts tuple of arguments to a list of dicts
    
    Parameters
    ----------
    names: list-like (optional), default None
        given names for each of the arguments in the tuple
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

def build_vset(name: str, obj, param_dict=None, *args, reps: int = 1,
               is_async: bool = False, output_matching: bool = False,
               lazy: bool = False, cache_dir: str = None, verbose: bool = True,
               tracking_dir: str = None, **kwargs) -> Vset:
    """Builds a Vset by currying callable obj with all combinations of parameters in param_dict.

    Parameters
    ----------
    name: str
        a name for the output Vset
    obj: callable
        a callable to use as the base for Vfuncs in the output Vset
    param_dict: dict[str, list]
        keys are obj kwarg names and values in the dict are lists of params to try
    *args
        additional fixed arguments to pass to obj
    reps: int (optional)
        the number of times to repeat the obj in the output Vset's modules for
        each combination of params in param_dict
    is_async: bool (optional)
        if True, modules are computed asynchronously
    output_matching: bool (optional)
        if True, then output keys from Vset will be matched when used
        in other Vsets
    cache_dir: str (optional)
        if provided, do caching and use cache_dir as the data store for
        joblib.Memory
    verbose : bool (optional)
        if True, modules are named with param_dict items as tuples of str("param_name=param_val")
    tracking_dir: str (optional)
        if provided, use the mlflow.tracking api to log outputs as metrics
        with params determined by input keys
    **kwargs
        additional fixed keyword arguments to pass to obj

    Returns
    -------
    new_vset : Vset
    """
    if param_dict is None:
        param_dict = {}
    assert callable(obj), 'obj must be callable'

    vfuncs = []
    vkeys = []

    # TODO: better way to check this?
    # check if obj is a class
    instantiate = isinstance(obj, type)

    param_names = list(param_dict.keys())
    param_lists = list(param_dict.values())
    kwargs_tuples = product(*param_lists)
    for tup in kwargs_tuples:
        kwargs_dict = {}
        vkey_tup = ()
        for param_name, param_val in zip(param_names, tup):
            kwargs_dict[param_name] = param_val
            vkey_tup += (f'{param_name}={param_val}', )
        # add additional fixed kwargs to kwargs_dict
        for k, v in kwargs.items():
            kwargs_dict[k] = v
        for i in range(reps):
            # add module key to vkeys
            if reps > 1:
                vkeys.append((f'rep={i}', ) + vkey_tup)
            else:
                vkeys.append(vkey_tup)
            if instantiate:
                # instantiate obj
                vfuncs.append(Vfunc(module=obj(*args, **kwargs_dict), name=str(vkey_tup)))
            else:
                # use partial to wrap obj
                vfuncs.append(Vfunc(module=partial(obj, *args, **kwargs_dict), name=str(vkey_tup)))
    if not verbose or (len(param_dict) == 0 and reps == 1):
        vkeys = None
    return Vset(name, vfuncs, is_async=is_async, module_keys=vkeys,
                output_matching=output_matching, lazy=lazy,
                cache_dir=cache_dir, tracking_dir=tracking_dir)


def filter_vset_by_metric(metric_dict: dict, vset: Vset, *vsets: Vset, n_keep: int = 1,
                          bigger_is_better: bool = True, filter_on=None,
                          group: bool = False) -> Union[Vset, list]:
    """Returns a new Vset by filtering `vset.modules` based on values in filter_dict.

    Parameters
    ----------
    metric_dict: dict
        output from a Vset, typically with metrics or other numeric values to use when
        filtering `vset.modules`
    vset: Vset
        a Vsets
    *vsets: Vset
        zero or more additional Vsets
    n_keep: int (optional)
        number of entries to keep from `vset.modules`
    bigger_is_better: bool (optional)
        if True, then the top `n_keep` largest values are retained
    filter_on: list[str] (optional)
        if there are multiple metrics in `metric_dict`, you can specify a subset
        to consider
    group: bool (optional)
        if True, average metrics after grouping values in `metric_dict` by the
        input Vset names

    Returns
    -------
    *new_vset : Vset
        Copies of the input Vsets but with Vfuncs filtered based on metrics
    """
    if filter_on is None:
        filter_on = []
    df = dict_to_df(metric_dict)
    vsets = [vset, *vsets]
    vset_names = []
    for vset in vsets:
        if vset.name not in df.columns:
            raise ValueError(f'{vset.name} should be one of the columns of dict_to_df(metric_dict)')
        vset_names.append(vset.name)
    if len(filter_on) > 0:
        filter_col = list(metric_dict.keys())[0][-1].origin
        df = df[df[filter_col].isin(filter_on)]
    if group:
        df = df.groupby(by=vset_names, as_index=False).mean()
    if bigger_is_better:
        df = df.sort_values(by='out', ascending=False)
    else:
        df = df.sort_values(by='out')
    df = df.iloc[0:n_keep]
    for i, vset in enumerate(vsets):
        vfuncs = vset.modules
        vfunc_filter = [str(name) for name in df[vset.name].to_numpy()]
        new_vfuncs = {k: v for k, v in vfuncs.items() if str(v.name) in vfunc_filter}
        new_vset = Vset('filtered_' + vset.name, new_vfuncs, is_async=vset._async,
                        output_matching=vset._output_matching, lazy=vset._lazy,
                        cache_dir=vset._cache_dir, tracking_dir=vset._tracking_dir)
        setattr(new_vset, FILTER_PREV_KEY, (metric_dict[PREV_KEY], vset,))
        setattr(new_vset, PREV_KEY, getattr(new_vset, FILTER_PREV_KEY))
        vsets[i] = new_vset
    if len(vsets) == 1:
        return vsets[0]
    else:
        return vsets

def perturbation_stats(data: Union[pd.DataFrame, dict], *group_by: str, wrt: str='out',
                       func=None, prefix: str=None, split: bool=False):
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
    mean_or_std = type(func) is list and 'mean' in func or 'std' in func
    list_or_ndarray = type(df[wrt].iloc[0]) in [list, np.ndarray]
    if mean_or_std and list_or_ndarray:
        dfs = [gb.get_group(grp) for grp in gb.groups]
        wrt_arrays = [np.stack(d.tolist()) for d in dfs]
        n_cols = wrt_arrays[0].shape[1]
        df_out = pd.DataFrame(gb.agg('count'))
        df_out.columns = [f'{prefix}-count']
        if 'mean' in func:
            if split:
                col_means = [arr.mean(axis=0) for arr in wrt_arrays]
                col_names = [f'{prefix}{i}-mean' for i in range(n_cols)]
                wrt_means = pd.DataFrame(col_means, columns=col_names,
                                         index=gb.groups.keys())
            else:
                col_means = [{f'{prefix}-mean': arr.mean(axis=0)} for arr in wrt_arrays]
                wrt_means = pd.DataFrame(col_means, index = gb.groups.keys())
            wrt_means.index.names = df_out.index.names
            df_out = df_out.join(wrt_means)
        if 'std' in func:
            if split:
                col_stds = [arr.std(axis=0, ddof=1) for arr in wrt_arrays]
                col_names = [f'{prefix}{i}-std' for i in range(n_cols)]
                wrt_stds = pd.DataFrame(col_stds, columns=col_names,
                                        index=gb.groups.keys())
            else:
                col_stds = [{f'{prefix}-std': arr.std(axis=0, ddof=1)} for arr in wrt_arrays]
                wrt_stds = pd.DataFrame(col_stds, index = gb.groups.keys())
            wrt_stds.index.names = df_out.index.names
            df_out = df_out.join(wrt_stds)
        if not 'count' in func:
            df_out = df_out.drop(f'{prefix}-count')
    else:
        df_out = gb.agg(func)
    df_out = df_out.reindex(sorted(df_out.columns), axis=1)
    df_out.reset_index(inplace=True)
    if len(group_by) > 0:
        return df_out.sort_values(group_by[0])
    return df_out

def cum_acc_by_uncertainty(mean_preds, std_preds, true_labels):
    """Returns uncertainty and cumulative accuracy for grouped class predictions,
    sorted in increasing order of uncertainty

    Params
    ------
    mean_preds: dict
        mean predictions, output from Vset.predict_with_uncertainties
    std_preds: dict
        std predictions, output from Vset.predict_with_uncertainties
    true_labels: dict or list-like

    TODO: generalize to multi-class classification
    """
    assert dict_keys(mean_preds) == dict_keys(std_preds), \
        "mean_preds and std_preds must share the same keys"
    # match predictions on keys
    paired_preds = [[d[k] for d in [mean_preds, std_preds]] for k in dict_keys(mean_preds)]
    mean_preds, std_preds = (np.array(p)[:,:,1] for p in zip(*paired_preds))
    if isinstance(true_labels, dict):
        true_labels = dict_data(true_labels)
        assert len(true_labels) == 1, 'true_labels should have a single 1D vector entry'
        true_labels = true_labels[0]
    n_obs = len(mean_preds[0])
    assert len(true_labels) == n_obs, \
        f'true_labels has {len(true_labels)} obs. but should have same as predictions ({n_obs})'
    sorted_idx = np.argsort(std_preds, axis=1)
    correct_labels = np.take_along_axis(np.around(mean_preds) - true_labels == 0, sorted_idx, 1)
    uncertainty = np.take_along_axis(std_preds, sorted_idx, 1)
    cum_acc = np.cumsum(correct_labels, axis=1) / range(1, n_obs+1)
    return uncertainty, cum_acc, sorted_idx