"""Set of modules to be parallelized over in a pipeline.
Function arguments are each a list
"""
from copy import deepcopy

import numpy as np
import joblib
import ray

from mlflow.tracking import MlflowClient

from vflow.subkey import Subkey
from vflow.utils import apply_modules, combine_dicts, dict_to_df, perturbation_stats, sep_dicts, \
    PREV_KEY
from vflow.vfunc import Vfunc, AsyncModule


FILTER_PREV_KEY = '__filter_prev__'


class Vset:

    def __init__(self, name: str, modules, module_keys: list = None,
                 is_async: bool = False, output_matching: bool = False,
                 lazy: bool = False, cache_dir: str = None,
                 tracking_dir: str = None):
        """
        Parameters
        ----------
        name: str
            Name of this Vset.
        modules: list or dict
            Dictionary of functions that we want to associate with
        module_keys: list (optional)
            List of names corresponding to each module
        is_async: bool (optional)
            If True, `modules` are computed asynchronously
        output_matching: bool (optional)
            If True, then output keys from this Vset will be matched when used
            in other Vsets
        lazy: bool (optional)
            If True, then modules are evaluated lazily, i.e. outputs are `vset.vfunc.VfuncPromise`
        cache_dir: str (optional)
            If provided, do caching and use `cache_dir` as the data store for
            `joblib.Memory`.
        tracking_dir: str (optional)
            If provided, use the `mlflow.tracking` api to log outputs as metrics
            with params determined by input keys.

        .. todo:: include prev and next and change functions to include that.
        """
        self.name = name
        self._fitted = False
        self.out = None  # outputs
        self._async = is_async
        self._output_matching = output_matching
        self._lazy = lazy
        self._cache_dir = cache_dir
        self._memory = joblib.Memory(self._cache_dir)
        if tracking_dir is not None:
            self._mlflow = MlflowClient(tracking_uri=tracking_dir)
            experiment = self._mlflow.get_experiment_by_name(name=self.name)
            if experiment is None:
                self._exp_id = self._mlflow.create_experiment(name=self.name)
            else:
                self._exp_id = experiment.experiment_id
        else:
            self._mlflow = None
        # check if any of the modules are AsyncModules
        # if so, we'll make then all AsyncModules later on
        if not self._async and np.any([isinstance(mod, AsyncModule) for mod in modules]):
            self._async = True
        if isinstance(modules, dict):
            self.modules = modules
        elif isinstance(modules, list):
            if module_keys is not None:
                assert isinstance(module_keys, list), 'modules passed as list but module_names is not a list'
                assert len(modules) == len(
                    module_keys), 'modules list and module_names list do not have the same length'
                # TODO: how best to handle tuple subkeys?
                module_keys = [(self.__create_subkey(k),) for k in module_keys]
            else:
                module_keys = [(self.__create_subkey(f'{name}_{i}'),) for i in range(len(modules))]
            # convert module keys to singleton tuples
            self.modules = dict(zip(module_keys, modules))
        # if needed, wrap the modules in the Vfunc or AsyncModule class
        for k, v in self.modules.items():
            if self._async:
                if not isinstance(v, AsyncModule):
                    self.modules[k] = AsyncModule(k[0], v)
            elif not isinstance(v, Vfunc):
                self.modules[k] = Vfunc(k[0], v)

    def _apply_func(self, *args, out_dict: dict = None):
        """Apply functions in out_dict to combined args dict

        Optionally logs output Subkeys and values as params and metrics using
        `mlflow.tracking` if this Vset has a `_tracking_dir`.

        Parameters
        ----------
        *args: dict
            Takes multiple dicts and combines them into one.
            Then runs modules on each item in combined dict.
        out_dict: dict (optional), default None
            The dictionary to pass to the matching function. If None, defaults to self.modules.

        Returns
        -------
        out_dict: dict
            Dictionary with items being determined by functions in module set.
            Functions and input dictionaries are currently matched using a cartesian matching format.

        Examples
        --------
        >>> modules, data = {LR : logistic}, {train_1 : [X1,y1], train2 : [X2,y2]}
        {(train_1, LR) : fitted logistic, (train_2, LR) :  fitted logistic}
        """
        if out_dict is None:
            out_dict = deepcopy(self.modules)

        apply_func_cached = self._memory.cache(_apply_func_cached)
        out_dict = apply_func_cached(out_dict, self._async, self._lazy, *args)

        prev = tuple()
        for arg in args:
            if PREV_KEY in arg:
                prev += (arg[PREV_KEY],)
        out_dict[PREV_KEY] = (self,) + prev

        if self._mlflow is not None:
            run_dict = {}
            # log subkeys as params and value as metric
            for k, v in out_dict.items():
                if k == PREV_KEY:
                    continue
                origins = np.array([subk.origin for subk in k])
                # ignore init origins and the last origin (this Vset)
                param_idx = [
                    i for i in range(len(k[:-1])) if origins[i] != 'init'
                ]
                # get or create mlflow run
                run_dict_key = tuple(subk.value for subk in k[:-1])
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
                            self._mlflow.log_param(
                                run_id, param_name, subkey.value
                            )
                self._mlflow.log_metric(run_id, k[-1].value, v)
        return out_dict

    def fit(self, *args):
        """Fits to args using `_apply_func`
        """
        out_dict = {}
        for k, v in self.modules.items():
            out_dict[k] = v.fit
        self.out = self._apply_func(*args, out_dict=out_dict)
        prev = self.out[PREV_KEY][1:]
        if hasattr(self, FILTER_PREV_KEY):
            prev = getattr(self, FILTER_PREV_KEY) + prev
        setattr(self, PREV_KEY, prev)
        self._fitted = True
        return self

    def fit_transform(self, *args):
        """Fits to args and transforms only the first arg.
        """
        return self.fit(*args).transform(args[0])

    def transform(self, *args):
        """Transforms args using `_apply_func`
        """
        if not self._fitted:
            raise AttributeError('Please fit the Vset object before calling the transform method.')
        out_dict = {}
        for k, v in self.out.items():
            if hasattr(v, 'transform'):
                out_dict[k] = v.transform
        return self._apply_func(*args, out_dict=out_dict)

    def predict(self, *args, with_uncertainty: bool=False, group_by: list=None):
        """Predicts args using `_apply_func`
        """
        if not self._fitted:
            raise AttributeError('Please fit the Vset object before calling predict.')
        pred_dict = {}
        for k, v in self.out.items():
            if hasattr(v, 'predict'):
                pred_dict[k] = v.predict
        preds = self._apply_func(*args, out_dict=pred_dict)
        if with_uncertainty:
            return prediction_uncertainty(preds, group_by)
        return preds

    def predict_proba(self, *args, with_uncertainty: bool=False, group_by: list=None):
        """Calls predict_proba on args using `_apply_func`
        """
        if not self._fitted:
            raise AttributeError('Please fit the Vset object before calling predict_proba.')
        pred_dict = {}
        for k, v in self.out.items():
            if hasattr(v, 'predict_proba'):
                pred_dict[k] = v.predict_proba
        preds = self._apply_func(*args, out_dict=pred_dict)
        if with_uncertainty:
            return prediction_uncertainty(preds, group_by)
        return preds

    def evaluate(self, *args):
        """Combines dicts before calling `_apply_func`
        """
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
        """Accesses ith item in the module set
        """
        return self.modules[i]

    def __contains__(self, key):
        """Returns true if modules is a dict and key is one of its keys
        """
        if isinstance(self.modules, dict):
            return key in self.modules.keys()
        return False

    def keys(self):
        """Returns Vset module keys
        """
        if isinstance(self.modules, dict):
            return self.modules.keys()
        return {}.keys()

    def __len__(self):
        return len(self.modules)

    def __str__(self):
        return 'Vset(' + self.name + ')'

    def __create_subkey(self, value):
        """Helper function to construct `Subkey` with
        this Vset determining origin and output_matching
        """
        return Subkey(value, self.name, self._output_matching)


def _apply_func_cached(out_dict: dict, is_async: bool, lazy: bool, *args):
    """
    Params
    ------
    *args: dict
        Takes multiple dicts and combines them into one.
        Then runs modules on each item in combined dict.
    out_dict: dict
        The dictionary to pass to the matching function.
    is_async: bool
        If True, outputs are computed asynchronously.
    lazy: bool
        If True, outputs are evaluated lazily, i.e. outputs are `VfuncPromise`.

    Returns
    -------
    out_dict: dict
        Dictionary with items being determined by functions in module set.
        Functions and input dictionaries are currently matched using cartesian matching format.
    """
    for in_dict in args:
        if not isinstance(in_dict, dict):
            raise Exception('Need to run init_args before calling module_set!')

    data_dict = combine_dicts(*args)
    out_dict = apply_modules(out_dict, data_dict, lazy)

    if is_async and not lazy:
        out_keys = list(out_dict.keys())
        out_vals = ray.get(list(out_dict.values()))
        out_dict = dict(zip(out_keys, out_vals))

    return out_dict


def prediction_uncertainty(preds, group_by: list=None):
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
        group_by = ['index']
    else:
        preds_stats = perturbation_stats(preds_df, *group_by)
    origins = preds_stats[group_by].columns
    keys = preds_stats[group_by].to_numpy()
    # wrap subkey values in Subkey
    keys = [tuple(Subkey(sk, origins[idx]) for idx, sk in enumerate(x)) for x in keys]
    mean_dict = dict(zip(keys, preds_stats['out-mean']))
    std_dict = dict(zip(keys, preds_stats['out-std']))
    # add PREV_KEY to out dicts
    mean_dict[PREV_KEY] = preds[PREV_KEY]
    std_dict[PREV_KEY] = preds[PREV_KEY]
    return mean_dict, std_dict, preds_stats
