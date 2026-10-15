"""User-facing helper functions included at import vflow
"""

import inspect
from functools import partial
from itertools import product
from typing import Union

import mlflow
import numpy as np

from vflow.utils import dict_data, dict_keys, dict_to_df
from vflow.vfunc import Vfunc
from vflow.vset import FILTER_PREV_KEY, PREV_KEY, Subkey, Vset


def init_args(args_tuple: Union[tuple, list], names=None):
    """Wraps each argument in a dict that Vsets can take as input.

    Parameters
    ----------
    args_tuple: tuple or list
        The data, e.g. the output of `train_test_split`.
    names: list-like (optional), default None
        A name for each argument, used to label outputs; defaults to "start".

    Returns
    -------
    list of dict
        One dict per argument, `{(Subkey(name, "init"),): arg, PREV_KEY: ("init",)}`.
    """
    if names is None:
        names = ["start"] * len(args_tuple)
    elif len(names) != len(args_tuple):
        raise ValueError("names should be same length as args_tuple")
    return [
        {(Subkey(name, "init"),): arg, PREV_KEY: ("init",)}
        for name, arg in zip(names, args_tuple)
    ]


def _key_origin(vset: Vset) -> str:
    """The origin of a Vset's vfunc keys, i.e. the column its outputs get in
    dict_to_df: the Vset's name, or the original name for a filtered Vset."""
    for k in vset.vfuncs:
        return k[0].origin
    return vset.name


def _func_name(f) -> str:
    """A readable name for `f`: a function's __name__, the wrapped function
    and its bound arguments for a functools.partial (so partials of one
    function get distinct names), the class name for a callable object."""
    if isinstance(f, partial):
        bound = [repr(a) for a in f.args]
        bound += [f"{k}={v!r}" for k, v in f.keywords.items()]
        return f"{_func_name(f.func)}({', '.join(bound)})"
    return getattr(f, "__name__", None) or type(f).__name__


def _same_config(a, b) -> bool:
    """Whether two (func, param values) pairs from build_vset are the same."""
    (fa, va), (fb, vb) = a, b
    if fa is not fb or len(va) != len(vb):
        return False
    for x, y in zip(va, vb):
        if x is y:
            continue
        if type(x) is not type(y):
            return False
        try:
            if not bool(x == y):
                return False
        except (TypeError, ValueError):  # e.g. arrays, whose == is elementwise
            return False
    return True


def _accepts_random_state(f) -> bool:
    """Whether `f` (a function, partial, class or callable object) can take a
    `random_state` keyword argument. Returns True when it cannot be told."""
    try:
        sig = inspect.signature(f)
    except (TypeError, ValueError):
        return True
    params = sig.parameters.values()
    return "random_state" in sig.parameters or any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in params
    )


def build_vset(
    name: str,
    func,
    param_dict=None,
    reps: int = 1,
    seed: int = None,
    is_async: bool = False,
    output_matching: bool = False,
    lazy: bool = False,
    cache_dir: str = None,
    tracking_dir: str = None,
    **kwargs,
) -> Vset:
    """Builds a new Vset by currying or instantiating callable `func` with all
    combinations of parameters in `param_dict` and optional additional `**kwargs`.
    If `func` and `param_dict` are lists, then the ith entry of `func` will be
    curried with ith entry of `param_dict`. If only one of `func` or `param_dict`
    is a list, the same `func`/`param_dict` will be curried for all entries in the
    list. Vfuncs are named with `param_dict` items as tuples of
    str("param_name=param_val").

    Parameters
    ----------
    name : str
        A name for the output Vset.
    func : callable or list[callable]
        A callable to use as the base for Vfuncs in the output Vset. Can also be
        a class object, in which case the class is immediately instantiated with
        the parameter combinations from `param_dict`. Can also be a list of
        callables, where the ith entry corresponds to `param_dict` or the ith
        entry of `param_dict` (if `param_dict` is a list).
    param_dict : dict[str, list] or list[dict[str, list]], optional
        A dict with string keys corresponding to argument names of `func` and
        entries which are lists of values to pass to `func` at run time (or when
        instantiating `func` if it's a class object). Can also be a list of
        dicts, where the ith dict entry corresponds to `func` or the ith entry
        of `func` (if `func` is a list). If no parameters are required for the
        ith function, the ith entry of `param_dict` can be `None`.
    reps : int, optional
        The number of times to repeat `func` in the output Vset's vfuncs for
        each combination of the parameters in `param_dict`.
    seed : int, optional
        Makes randomized vfuncs reproducible. When given, every vfunc gets its
        own `random_state`, derived from `seed` (with numpy's SeedSequence) and
        passed to `func` as a keyword argument. So `reps` copies of e.g.
        `sklearn.utils.resample` draw different samples, and building the Vset
        again with the same `seed` and arguments draws the same ones. `func`
        must accept a `random_state` argument, and `random_state` must not also
        be set in `param_dict` or `**kwargs`. If None (the default), nothing is
        seeded and repetitions use whatever randomness `func` uses by default.
    is_async : bool, optional
        If True, vfuncs are computed asynchronously.
    output_matching : bool, optional
        If True, then output keys from Vset will be matched when used in other
        Vsets.
    lazy : bool, optional
        If True, outputs are computed only when used (they are `VfuncPromise`s).
    cache_dir : str, optional
        If provided, do caching and use `cache_dir` as the data store for
        joblib.Memory.
    tracking_dir : str, optional
        If provided, use the mlflow.tracking API to log outputs as metrics with
        parameters determined by input keys.
    **kwargs
        Additional fixed keyword arguments to pass to `func`.

    Returns
    -------
    new_vset : vflow.vset.Vset

    """
    # pair each func with its param_dict; a single func or param_dict is
    # shared by every entry of the other's list
    n = len(func) if isinstance(func, list) else 1
    if isinstance(param_dict, list):
        if isinstance(func, list) and len(param_dict) != n:
            raise ValueError("list of param_dicts must be same length as list of funcs")
        n = len(param_dict)
    funcs = func if isinstance(func, list) else [func] * n
    given = param_dict if isinstance(param_dict, list) else [param_dict] * n
    param_dicts = [p if p is not None else {} for p in given]
    for f in funcs:
        if not callable(f):
            raise TypeError(f"func must be callable, got {f!r}")

    seeds = _seeds(seed, funcs, param_dicts, reps, kwargs) if seed is not None else None
    vfuncs = []
    vkeys = []
    # (func, param values) behind each key, to tell a repeated configuration
    # from distinct ones that happen to print the same
    configs = []
    for f, params in zip(funcs, param_dicts):
        for values in product(*params.values()):
            f_kwargs = {**dict(zip(params, values)), **kwargs}
            vkey = (f"func={_func_name(f)}",) + tuple(
                f"{k}={v}" for k, v in zip(params, values)
            )
            for i in range(reps):
                vkeys.append((f"rep={i}",) + vkey if reps > 1 else vkey)
                configs.append((f, values))
                rep_kwargs = dict(f_kwargs)
                if seeds is not None:
                    rep_kwargs["random_state"] = next(seeds)
                # a class is instantiated now; a function is called later
                vfunc = (
                    f(**rep_kwargs) if isinstance(f, type) else partial(f, **rep_kwargs)
                )
                vfuncs.append(Vfunc(vfunc=vfunc, name=str(vkey)))

    if all(p is None for p in given) and reps == 1:
        vkeys = None  # Vset names the vfuncs {name}_0, {name}_1, ...
    elif len(set(vkeys)) < len(vkeys):
        vkeys, vfuncs = _merge_repeated_configs(name, vkeys, vfuncs, configs)

    return Vset(
        name,
        vfuncs,
        is_async=is_async,
        vfunc_keys=vkeys,
        output_matching=output_matching,
        lazy=lazy,
        cache_dir=cache_dir,
        tracking_dir=tracking_dir,
    )


def _seeds(seed, funcs, param_dicts, reps, kwargs):
    """An iterator of one independent, reproducible random_state per vfunc
    that build_vset will make, checking `seed` can be used."""
    if "random_state" in kwargs or any("random_state" in p for p in param_dicts):
        raise ValueError(
            "pass either `seed` or `random_state`, not both: `seed` sets a "
            "different random_state for every vfunc"
        )
    for f in funcs:
        if not _accepts_random_state(f):
            raise ValueError(
                f"`seed` needs `func` to accept a `random_state` argument, "
                f"and {_func_name(f)} does not"
            )
    n_vfuncs = reps * sum(
        int(np.prod([len(v) for v in p.values()])) for p in param_dicts
    )
    return iter(int(s) for s in np.random.SeedSequence(seed).generate_state(n_vfuncs))


def _merge_repeated_configs(name, vkeys, vfuncs, configs):
    """Overlapping param_dicts repeat a configuration; keep one vfunc per key
    (the last, as a dict would), in first-seen order. Distinct configurations
    with the same key (two lambdas, partials of one function, 1 and "1") would
    be silently dropped, so refuse those."""
    first_config = {}
    for key, config in zip(vkeys, configs):
        seen = first_config.setdefault(key, config)
        if not _same_config(seen, config):
            raise ValueError(
                f"build_vset({name!r}): different functions or parameter "
                f"values share the key {key}; give the functions distinct "
                "names (e.g. with def instead of lambda or partial) and use "
                "parameter values that print differently"
            )
    merged = dict(zip(vkeys, vfuncs))
    return list(merged.keys()), list(merged.values())


def filter_vset_by_metric(
    metric_dict: dict,
    vset: Vset,
    *vsets: Vset,
    n_keep: int = 1,
    bigger_is_better: bool = True,
    filter_on=None,
    group: bool = False,
) -> Union[Vset, list]:
    """Returns a new Vset by filtering `vset.vfuncs` based on values in
    `metric_dict`.

    Parameters
    ----------
    metric_dict: dict
        output from a Vset, typically with metrics or other numeric values to use when
        filtering `vset.vfuncs`
    vset: Vset
        the Vset to filter
    *vsets: Vset
        zero or more additional Vsets
    n_keep: int (optional)
        number of entries to keep from `vset.vfuncs`
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
    Vset or list of Vset
        A copy of `vset` (a list of copies, given `*vsets`) keeping only the
        vfuncs of the best `n_keep` results
    """
    df = dict_to_df(metric_dict)
    vsets = [vset, *vsets]
    vset_names = []
    for vset_i in vsets:
        # outputs are labelled by the origin of the vfunc keys, which stays the
        # original Vset's name after filtering (a filtered Vset is renamed)
        col = _key_origin(vset_i)
        if col not in df.columns:
            raise ValueError(
                f"{col} should be one of the columns of dict_to_df(metric_dict)"
            )
        vset_names.append(col)
    if filter_on:
        filter_col = [k for k in metric_dict if k != PREV_KEY][0][-1].origin
        df = df[df[filter_col].isin(filter_on)]
    if group:
        df = df.groupby(by=vset_names, as_index=False).mean(numeric_only=True)
    # stable sort: among tied values, the earlier row wins, every time
    df = df.sort_values(by="out", ascending=not bigger_is_better, kind="stable")
    # n_keep counts distinct vfunc combinations, not metric rows: one vfunc can
    # top several rows when it was run on several upstream perturbations
    df = df.drop_duplicates(subset=vset_names).iloc[0:n_keep]
    new_vsets = []
    for vset_i, col in zip(vsets, vset_names):
        keep = list(df[col])
        # match on the key the outputs are labelled with, not on Vfunc.name
        new_vfuncs = {k: v for k, v in vset_i.vfuncs.items() if k[0].value in keep}
        tracking_dir = None if vset_i._mlflow is None else mlflow.get_tracking_uri()
        new_vset = Vset(
            "filtered_" + vset_i.name,
            new_vfuncs,
            is_async=vset_i._async,
            output_matching=vset_i._output_matching,
            lazy=vset_i._lazy,
            cache_dir=vset_i._cache_dir,
            tracking_dir=tracking_dir,
        )
        # lineage for build_graph: the metrics that chose it, and the original
        prev = (metric_dict[PREV_KEY], vset_i)
        setattr(new_vset, FILTER_PREV_KEY, prev)
        setattr(new_vset, PREV_KEY, prev)
        new_vsets.append(new_vset)
    return new_vsets[0] if len(new_vsets) == 1 else new_vsets


def cum_acc_by_uncertainty(mean_preds, std_preds, true_labels):
    """Returns uncertainty and cumulative accuracy for grouped class predictions,
    sorted in increasing order of uncertainty

    Parameters
    ----------
    mean_preds: dict
        mean class probabilities per group, from
        Vset.predict_proba(..., with_uncertainty=True)
    std_preds: dict
        std of class probabilities per group, from the same call
    true_labels: dict or list-like

    TODO: generalize to multi-class classification (this uses the
    probability of class 1)
    """
    keys = dict_keys(mean_preds)
    if keys != dict_keys(std_preds):
        raise ValueError("mean_preds and std_preds must share the same keys")
    # (n_groups, n_obs, n_classes), matched on keys
    mean_arr = np.array([mean_preds[k] for k in keys])
    std_arr = np.array([std_preds[k] for k in keys])
    if mean_arr.ndim != 3:
        raise ValueError(
            "cum_acc_by_uncertainty needs class probabilities of shape "
            "(n_obs, n_classes) per group, as from "
            "Vset.predict_proba(..., with_uncertainty=True); got predictions "
            f"of shape {mean_arr.shape[1:]}"
        )
    mean_pos, std_pos = mean_arr[:, :, 1], std_arr[:, :, 1]
    if isinstance(true_labels, dict):
        true_labels = dict_data(true_labels)
        if len(true_labels) != 1:
            raise ValueError("true_labels should have a single 1D vector entry")
        true_labels = true_labels[0]
    n_obs = mean_pos.shape[1]
    if len(true_labels) != n_obs:
        raise ValueError(
            f"true_labels has {len(true_labels)} obs. but should have same as "
            f"predictions ({n_obs})"
        )
    sorted_idx = np.argsort(std_pos, axis=1)
    correct_labels = np.take_along_axis(
        np.around(mean_pos) - true_labels == 0, sorted_idx, 1
    )
    uncertainty = np.take_along_axis(std_pos, sorted_idx, 1)
    cum_acc = np.cumsum(correct_labels, axis=1) / range(1, n_obs + 1)
    return uncertainty, cum_acc, sorted_idx
