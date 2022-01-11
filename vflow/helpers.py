"""User-facing helper functions included at import vflow
"""
from functools import partial
from itertools import product
from typing import Union

import mlflow
import numpy as np

from vflow.utils import dict_to_df, dict_keys, dict_data
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
    for i, _ in enumerate(args_tuple):
        output_dicts.append({
            (Subkey(names[i], 'init'),): args_tuple[i],
            PREV_KEY: ('init',),
        })
    return output_dicts


def build_vset(name: str, obj, *args, param_dict=None, reps: int = 1,
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

    kwargs_tuples = product(*list(param_dict.values()))
    for tup in kwargs_tuples:
        kwargs_dict = {}
        vkey_tup = ()
        for param_name, param_val in zip(list(param_dict.keys()), tup):
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
            # check if obj is a class
            if isinstance(obj, type):
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
    for vset_i in vsets:
        if vset_i.name not in df.columns:
            raise ValueError((f'{vset_i.name} should be one '
                              'of the columns of dict_to_df(metric_dict)'))
        vset_names.append(vset_i.name)
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
    for i, vset_i in enumerate(vsets):
        vfuncs = vset_i.modules
        vfunc_filter = [str(name) for name in df[vset_i.name].to_numpy()]
        new_vfuncs = {k: v for k, v in vfuncs.items() if str(v.name) in vfunc_filter}
        tracking_dir = None if vset_i._mlflow is None else mlflow.get_tracking_uri()
        new_vset = Vset('filtered_' + vset_i.name, new_vfuncs, is_async=vset_i._async,
                        output_matching=vset_i._output_matching, lazy=vset_i._lazy,
                        cache_dir=vset_i._cache_dir, tracking_dir=tracking_dir)
        setattr(new_vset, FILTER_PREV_KEY, (metric_dict[PREV_KEY], vset_i,))
        setattr(new_vset, PREV_KEY, getattr(new_vset, FILTER_PREV_KEY))
        vsets[i] = new_vset
    if len(vsets) == 1:
        return vsets[0]
    return vsets


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
    paired_preds = [[d[k] for d in (mean_preds, std_preds)] for k in dict_keys(mean_preds)]
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
