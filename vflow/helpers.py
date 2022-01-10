"""User-facing helper functions included at import vflow
"""
from functools import partial
from itertools import product
from typing import Union

from vflow.convert import dict_to_df
from vflow.vfunc import Vfunc
from vflow.vset import Vset, PREV_KEY, FILTER_PREV_KEY


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
