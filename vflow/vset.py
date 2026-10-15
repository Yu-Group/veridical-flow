"""Set of vfuncs to be parallelized over in a pipeline.
Function arguments are each a list
"""

import hashlib
import inspect
import os
import re
import types
import warnings
from copy import deepcopy
from functools import partial

import joblib
import numpy as np
import ray
from mlflow.tracking import MlflowClient

try:
    import cloudpickle
except ImportError:  # ray always ships a copy
    from ray import cloudpickle

from vflow.subkey import Subkey
from vflow.utils import (
    PREV_KEY,
    apply_vfuncs,
    combine_dicts,
    dict_to_df,
    perturbation_stats,
    sep_dicts,
)
from vflow.vfunc import AsyncVfunc, Vfunc

FILTER_PREV_KEY = "__filter_prev__"


class Vset:
    def __init__(
        self,
        name: str,
        vfuncs,
        vfunc_keys: list = None,
        is_async: bool = False,
        output_matching: bool = False,
        lazy: bool = False,
        cache_dir: str = None,
        tracking_dir: str = None,
    ):
        """
        Parameters
        ----------
        name: str
            Name of this Vset.
        vfuncs: list or dict
            Dictionary of functions that we want to associate with
        vfunc_keys: list (optional)
            List of names corresponding to each vfunc
        is_async: bool (optional)
            If True, `vfuncs` are computed asynchronously
        output_matching: bool (optional)
            If True, then output keys from this Vset will be matched when used
            in other Vsets
        lazy: bool (optional)
            If True, then vfuncs are evaluated lazily, i.e. outputs are `vset.vfunc.VfuncPromise`
        cache_dir: str (optional)
            If provided, do caching and use `cache_dir` as the data store for
            `joblib.Memory`.
        tracking_dir: str (optional)
            If provided, use the `mlflow.tracking` api to log outputs as metrics
            with params determined by input keys.

        """
        self.name = name
        self._fitted = False
        self.fitted_vfuncs = None  # outputs
        self._async = is_async
        self._output_matching = output_matching
        self._lazy = lazy
        self._cache_dir = cache_dir
        # verbose=0: by default joblib prints every call with its full arguments
        self._memory = joblib.Memory(self._cache_dir, verbose=0)
        if tracking_dir is not None:
            if "://" not in str(tracking_dir):
                # a plain directory: MLflow 3 needs an explicit opt-in for its
                # file-based tracking store (respects a value the user has set)
                os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
            self._mlflow = MlflowClient(tracking_uri=tracking_dir)
            experiment = self._mlflow.get_experiment_by_name(name=self.name)
            if experiment is None:
                self._exp_id = self._mlflow.create_experiment(name=self.name)
            else:
                self._exp_id = experiment.experiment_id
        else:
            self._mlflow = None
        if not isinstance(vfuncs, (dict, list)):
            raise TypeError(
                f"vfuncs must be a list or a dict, not {type(vfuncs).__name__}"
            )
        # check if any of the vfuncs are AsyncVfuncs
        # if so, we'll make then all AsyncVfuncs later on
        funcs = vfuncs.values() if isinstance(vfuncs, dict) else vfuncs
        if not self._async and any(isinstance(vf, AsyncVfunc) for vf in funcs):
            self._async = True
        if isinstance(vfuncs, dict):
            # keys may already be tuples of Subkeys (e.g. from another Vset);
            # plain keys like "LR" become this Vset's subkeys
            self.vfuncs = {
                (
                    k
                    if isinstance(k, tuple) and all(isinstance(s, Subkey) for s in k)
                    else (self.__create_subkey(k),)
                ): v
                for k, v in vfuncs.items()
            }
        else:
            if vfunc_keys is not None:
                assert isinstance(
                    vfunc_keys, list
                ), "vfuncs passed as list but vfunc_keys is not a list"
                assert len(vfuncs) == len(
                    vfunc_keys
                ), "vfuncs list and vfunc_keys list do not have the same length"
                # TODO: how best to handle tuple subkeys?
                vfunc_keys = [(self.__create_subkey(k),) for k in vfunc_keys]
                # compare as the dict below will (1 and 1.0 are one key)
                if len(dict.fromkeys(vfunc_keys)) < len(vfunc_keys):
                    raise ValueError(
                        f"vfunc_keys must be unique; duplicates would silently "
                        f"drop vfuncs: {[k[0].value for k in vfunc_keys]}"
                    )
            else:
                vfunc_keys = [
                    (self.__create_subkey(f"{name}_{i}"),) for i in range(len(vfuncs))
                ]
            # convert vfunc keys to singleton tuples
            self.vfuncs = dict(zip(vfunc_keys, vfuncs))
        # if needed, wrap the vfuncs in the Vfunc or AsyncVfunc class
        for k, v in self.vfuncs.items():
            if self._async:
                if not isinstance(v, AsyncVfunc):
                    self.vfuncs[k] = AsyncVfunc(k[0], v)
            elif not isinstance(v, Vfunc):
                self.vfuncs[k] = Vfunc(k[0], v)

    def _apply_func(self, *args, out_dict: dict = None):
        """Apply functions in out_dict to combined args dict

        Optionally logs output Subkeys and values as params and metrics using
        `mlflow.tracking` if this Vset has a `_tracking_dir`.

        Parameters
        ----------
        *args: dict
            Takes multiple dicts and combines them into one.
            Then runs vfuncs on each item in combined dict.
        out_dict: dict (optional), default None
            The dictionary to pass to the matching function. If None, defaults to self.vfuncs.

        Returns
        -------
        out_dict: dict
            Dictionary with items being determined by functions in vfunc set.
            Functions and input dictionaries are currently matched using a cartesian matching format.

        Examples
        --------
        >>> vfuncs, data = {LR : logistic}, {train_1 : [X1,y1], train2 : [X2,y2]}
        {(train_1, LR) : fitted logistic, (train_2, LR) :  fitted logistic}
        """
        if out_dict is None:
            out_dict = deepcopy(self.vfuncs)

        cache_key = None
        if self._cache_dir is not None:
            # joblib hashes functions by name, so redefining a function would
            # serve stale results, and it cannot hash lambdas at all. Key the
            # cache on the functions' code and the inputs instead.
            try:
                cache_key = _cache_key(out_dict, self._async, self._lazy, args)
            except Exception as e:  # noqa: BLE001 - fall back to no caching
                warnings.warn(
                    f"Vset {self.name!r}: could not fingerprint the inputs for "
                    f"caching ({type(e).__name__}: {e}); running without the cache"
                )
        if cache_key is None:
            out_dict = _apply_func_cached(None, out_dict, self._async, self._lazy, args)
        else:
            apply_func_cached = self._memory.cache(
                _apply_func_cached, ignore=["out_dict", "args"]
            )
            out_dict = apply_func_cached(
                cache_key, out_dict, self._async, self._lazy, args
            )

        prev = tuple()
        for arg in args:
            if PREV_KEY in arg:
                prev += (arg[PREV_KEY],)
        out_dict[PREV_KEY] = (self,) + prev

        if self._mlflow is not None:
            run_dict = {}
            try:
                # log subkeys as params and value as metric
                for k, v in out_dict.items():
                    # only numeric scalars are metrics; fitted models, predictions
                    # and lazy promises are not logged
                    if k == PREV_KEY or not _is_numeric_scalar(v):
                        continue
                    origins = np.array([subk.origin for subk in k])
                    # ignore init origins and the last origin (this Vset)
                    param_idx = [i for i in range(len(k[:-1])) if origins[i] != "init"]
                    # get or create mlflow run
                    # Subkeys compare by value and origin, so equal values
                    # from different Vsets get their own runs
                    run_dict_key = k[:-1]
                    if run_dict_key in run_dict:
                        run_id = run_dict[run_dict_key]
                    else:
                        run = self._mlflow.create_run(self._exp_id)
                        run_id = run.info.run_id
                        run_dict[run_dict_key] = run_id
                        # log params
                        for idx in param_idx:
                            subkey = k[idx]
                            param_name = subkey.origin
                            # check if the origin occurs multiple times
                            if np.sum(origins == param_name) > 1:
                                occurence = np.sum(origins[:idx] == param_name)
                                param_name = param_name + str(occurence)
                            self._mlflow.log_param(run_id, param_name, subkey.value)
                    self._mlflow.log_metric(run_id, _metric_name(k[-1].value), float(v))
            finally:
                # close the runs, even if logging fails, or they stay RUNNING
                for run_id in run_dict.values():
                    self._mlflow.set_terminated(run_id)
        return out_dict

    def fit(self, *args):
        """Fits to args using `_apply_func`"""
        out_dict = {}
        for k, v in self.vfuncs.items():
            out_dict[k] = v.fit
        self.fitted_vfuncs = self._apply_func(*args, out_dict=out_dict)
        prev = self.fitted_vfuncs[PREV_KEY][1:]
        if hasattr(self, FILTER_PREV_KEY):
            prev = getattr(self, FILTER_PREV_KEY) + prev
        setattr(self, PREV_KEY, prev)
        self._fitted = True
        return self

    def fit_transform(self, *args):
        """Fits to args and transforms only the first arg."""
        return self.fit(*args).transform(args[0])

    def transform(self, *args):
        """Transforms args using `_apply_func`"""
        if not self._fitted:
            raise AttributeError(
                "Please fit the Vset object before calling the transform method."
            )
        out_dict = {}
        for k, v in self.fitted_vfuncs.items():
            if hasattr(v, "transform"):
                out_dict[k] = v.transform
        return self._apply_func(*args, out_dict=out_dict)

    def predict(self, *args, with_uncertainty: bool = False, group_by: list = None):
        """Predicts args using `_apply_func`"""
        if not self._fitted:
            raise AttributeError("Please fit the Vset object before calling predict.")
        pred_dict = {}
        for k, v in self.fitted_vfuncs.items():
            if hasattr(v, "predict"):
                pred_dict[k] = v.predict
        self._require_method(pred_dict, "predict")
        preds = self._apply_func(*args, out_dict=pred_dict)
        if with_uncertainty:
            return prediction_uncertainty(preds, group_by)
        return preds

    def predict_proba(
        self, *args, with_uncertainty: bool = False, group_by: list = None
    ):
        """Calls predict_proba on args using `_apply_func`"""
        if not self._fitted:
            raise AttributeError(
                "Please fit the Vset object before calling predict_proba."
            )
        pred_dict = {}
        for k, v in self.fitted_vfuncs.items():
            if hasattr(v, "predict_proba"):
                pred_dict[k] = v.predict_proba
        self._require_method(pred_dict, "predict_proba")
        preds = self._apply_func(*args, out_dict=pred_dict)
        if with_uncertainty:
            return prediction_uncertainty(preds, group_by)
        return preds

    def evaluate(self, *args):
        """Combines dicts before calling `_apply_func`"""
        return self._apply_func(*args)

    def __call__(self, *args, n_out: int = None, keys=None, **kwargs):
        """Call args using `_apply_func`, optionally seperating
        output dictionary into `n_out` dictionaries with `keys`
        """
        if keys is None:
            keys = []
        if n_out is None:
            n_out = len(args)
        out_dict = self._apply_func(*args)
        if n_out == 1:
            return out_dict
        out_dicts = sep_dicts(out_dict, n_out=n_out, keys=keys)
        # add back prev
        prev = out_dict[PREV_KEY]
        for i in range(n_out):
            if n_out == len(args):
                out_dicts[i][PREV_KEY] = (prev[0],) + (prev[i + 1],)
            else:
                out_dicts[i][PREV_KEY] = prev
        return out_dicts

    def __getitem__(self, i):
        """Accesses ith item in the vfunc set"""
        return self.vfuncs[i]

    def __contains__(self, key):
        """Returns true if vfuncs is a dict and key is one of its keys"""
        if isinstance(self.vfuncs, dict):
            return key in self.vfuncs.keys()
        return False

    def keys(self):
        """Returns Vset vfunc keys"""
        if isinstance(self.vfuncs, dict):
            return self.vfuncs.keys()
        return {}.keys()

    def __len__(self):
        return len(self.vfuncs)

    def __str__(self):
        return "Vset(" + self.name + ")"

    def _require_method(self, method_dict, method):
        """Raise rather than silently return no outputs when no fitted vfunc
        has `method`."""
        if not method_dict:
            raise AttributeError(
                f"None of the fitted vfuncs in Vset {self.name!r} has a "
                f"{method!r} method."
            )

    def __create_subkey(self, value):
        """Helper function to construct `Subkey` with
        this Vset determining origin and output_matching
        """
        return Subkey(value, self.name, self._output_matching)


def _metric_name(value) -> str:
    """An MLflow metric name for a subkey value: build_vset keys are tuples
    like ('func=score', 'scale=1'), and MLflow allows only letters, digits,
    and _ - . / : and spaces."""
    if isinstance(value, tuple):
        value = "_".join(map(str, value))
    return re.sub(r"[^\w\-./: ]", "_", str(value))


def _is_numeric_scalar(v) -> bool:
    """Whether `v` can be logged as an MLflow metric."""
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(
        v, bool
    )


def _cache_key(out_dict: dict, is_async: bool, lazy: bool, args: tuple) -> str:
    """A fingerprint of the vfuncs in `out_dict` and the data in `args`.

    The data is hashed with joblib, which hashes by value: a cloudpickle of it
    would change with object identity (an output loaded back from the cache
    is an equal copy that shares objects differently), so downstream Vsets
    would miss the cache. The lineage entry (PREV_KEY) holds Vset objects, not
    data, and is left out. The vfuncs are pickled with cloudpickle, which
    pickles lambdas and functions defined in __main__ by value; functions from
    importable modules are pickled by reference, so their source is added.
    """
    data = tuple(
        {k: v for k, v in a.items() if k != PREV_KEY} if isinstance(a, dict) else a
        for a in args
    )
    try:
        data_hash = joblib.hash(data)
    except Exception:  # noqa: BLE001 - e.g. lazy promises holding lambdas
        data_hash = hashlib.sha256(cloudpickle.dumps(data)).hexdigest()
    h = hashlib.sha256(cloudpickle.dumps((out_dict, is_async, lazy)))
    h.update(data_hash.encode())
    for source in _function_sources(out_dict.values()):
        h.update(source.encode())
    return h.hexdigest()


def _function_sources(objs) -> list:
    """Source code of the Python functions reachable from `objs` (unwrapping
    Vfuncs, bound methods and partials), in a deterministic order."""
    sources, seen, stack = [], set(), list(objs)[::-1]
    while stack:
        obj = stack.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        if isinstance(obj, (Vfunc, AsyncVfunc)):
            stack.append(obj.vfunc)
        elif isinstance(obj, types.MethodType):
            stack += [obj.__self__, obj.__func__]
        elif isinstance(obj, partial):
            stack += [*obj.keywords.values(), *obj.args, obj.func]
        elif isinstance(obj, types.FunctionType):
            try:
                sources.append(inspect.getsource(obj))
            except (OSError, TypeError):  # no source file, e.g. in a REPL
                pass
    return sources


def _apply_func_cached(
    cache_key, out_dict: dict, is_async: bool, lazy: bool, args: tuple
):
    """
    Params
    ------
    cache_key: str or None
        Fingerprint of the vfuncs and inputs; the only argument joblib hashes
        when caching (out_dict and args are ignored by the cache).
    out_dict: dict
        The dictionary to pass to the matching function.
    is_async: bool
        If True, outputs are computed asynchronously.
    lazy: bool
        If True, outputs are evaluated lazily, i.e. outputs are `VfuncPromise`.
    args: tuple of dict
        Takes multiple dicts and combines them into one.
        Then runs vfuncs on each item in combined dict.

    Returns
    -------
    out_dict: dict
        Dictionary with items being determined by functions in vfunc set.
        Functions and input dictionaries are currently matched using cartesian matching format.
    """
    for in_dict in args:
        if not isinstance(in_dict, dict):
            raise Exception(
                "Run init_args on data before using it when calling a Vset!"
            )

    data_dict = combine_dicts(*args)
    out_dict = apply_vfuncs(out_dict, data_dict, lazy)

    if is_async and not lazy:
        out_keys = list(out_dict.keys())
        out_vals = list(out_dict.values())
        if isinstance(out_vals[0], ray.ObjectRef):
            out_vals = ray.get(out_vals)
        out_dict = dict(zip(out_keys, out_vals))

    return out_dict


def prediction_uncertainty(preds, group_by: list = None):
    """Returns the mean and std predictions conditional on group_by

    Params
    ------
    preds
        predictions as returned by Vset.predict or Vset.predict_proba
    group_by: list (optional), default None
        list of groups to compute statistics upon

    TODO: Wrap output dicts in dict wrapper::XXX
          Wrap subkeys in Subkey
          Fix default group_by when averaging over all predictions
    """
    preds_df = dict_to_df(preds)
    if group_by is None:
        # just average over all predictions
        preds_stats = perturbation_stats(preds_df)
        group_by = ["index"]
    else:
        preds_stats = perturbation_stats(preds_df, *group_by)
    origins = preds_stats[group_by].columns
    keys = preds_stats[group_by].to_numpy()
    # wrap subkey values in Subkey
    keys = [tuple(Subkey(sk, origins[idx]) for idx, sk in enumerate(x)) for x in keys]
    mean_dict = dict(zip(keys, preds_stats["out-mean"]))
    std_dict = dict(zip(keys, preds_stats["out-std"]))
    # add PREV_KEY to out dicts
    mean_dict[PREV_KEY] = preds[PREV_KEY]
    std_dict[PREV_KEY] = preds[PREV_KEY]
    return mean_dict, std_dict, preds_stats
