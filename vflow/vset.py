'''Set of modules to be parallelized over in a pipeline.
Function arguments are each a list
'''
PREV_KEY = '__prev__'

import numpy as np
import joblib
import ray

from  mlflow.tracking import MlflowClient

from vflow.convert import *
from vflow.vfunc import Vfunc, AsyncModule
from vflow.subkey import Subkey

class Vset:
    def __init__(self, name: str, modules, module_keys: list = None,
                 is_async: bool = False, output_matching: bool = False,
                 lazy: bool = False, cache_dir: str = None,
                 tracking_dir: str = None):
        '''
        todo: include prev and next and change functions to include that.
        Params
        -------
        name: str
            name of this moduleset
        modules: list or dict
            dictionary of functions that we want to associate with
        module_keys: list (optional)
            list of names corresponding to each module
        is_async: bool (optional)
            if True, modules are computed asynchronously
        output_matching: bool (optional)
            if True, then output keys from this Vset will be matched when used
            in other Vsets
        lazy: bool (optional)
            if True, then modules are evaluated lazily, i.e. outputs contain a
            promise
        cache_dir: str (optional)
            if provided, do caching and use cache_dir as the data store for
            joblib.Memory
        tracking_dir: str (optional)
            if provided, use the mlflow.tracking api to log outputs as metrics
            with params determined by input keys
        '''
        self.name = name
        self._fitted = False
        self.out = None  # outputs
        self._async = is_async
        self._output_matching = output_matching
        self._lazy = lazy
        self._memory = joblib.Memory(cache_dir)
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
        if type(modules) is dict:
            self.modules = modules
        elif type(modules) is list:
            if module_keys is not None:
                assert type(module_keys) is list, 'modules passed as list but module_names is not a list'
                assert len(modules) == len(
                    module_keys), 'modules list and module_names list do not have the same length'
                # TODO: add more checking of module_keys
                module_keys = [self.__create_subkey(k) if isinstance(k, tuple) else
                                (self.__create_subkey(k), ) for k in module_keys]
            else:
                module_keys = [(self.__create_subkey(f'{name}_{i}'), ) for i in range(len(modules))]
            # convert module keys to singleton tuples
            self.modules = dict(zip(module_keys, modules))
        # if needed, wrap the modules in the Vfunc or AsyncModule class
        for k, v in self.modules.items():
            if self._async:
                if not isinstance(v, AsyncModule):
                    self.modules[k] = AsyncModule(k[0], v)
            elif not isinstance(v, Vfunc):
                self.modules[k] = Vfunc(k[0], v)

    def _apply_func(self, out_dict: dict=None, *args):
        if out_dict is None:
            out_dict = deepcopy(self.modules)
        apply_func_cached = self._memory.cache(_apply_func_cached)
        data_dict, out_dict = apply_func_cached(
            out_dict, self._async, self._lazy, *args
        )
        if PREV_KEY in data_dict:
            self.__prev__ = data_dict[PREV_KEY]
        else:
            self.__prev__ = ('init', )
        if self._mlflow is not None:
            run_dict = {}
            # log subkeys as params and value as metric
            for k, v in out_dict.items():
                origins = np.array([subk.origin for subk in k])
                # ignore init origins and the last origin (this Vset)
                param_idx = [
                    i for i in range(len(k[:-1])) if origins[i] != 'init'
                ]
                # get or create mlflow run
                run_dict_key = tuple([subk.value for subk in k[:-1]])
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
                            occurence  = np.sum(origins[:idx] == param_name)
                            param_name = param_name + str(occurence)
                            self._mlflow.log_param(
                                run_id, param_name, subkey.value
                            )
                self._mlflow.log_metric(run_id, k[-1].value, v)
        out_dict[PREV_KEY] = (self,)
        return out_dict

    def fit(self, *args, **kwargs):
        '''
        '''
        if self._fitted:
            return self
        out_dict = {}
        for k, v in self.modules.items():
            out_dict[k] = v.fit
        self.out = self._apply_func(out_dict, *args)
        self._fitted = True
        return self

    def transform(self, *args, **kwargs):
        '''todo: fix this method
        '''
        results = []
        for out in self.output:
            result = out.transform(*args, **kwargs)
            results.append(result)
        return results

    def predict(self, *args, **kwargs):
        if not self._fitted:
            raise AttributeError('Please fit the Vset object before calling the predict method.')
        pred_dict = {}
        for k, v in self.out.items():
            if hasattr(v, 'predict'):
                pred_dict[k] = v.predict
        return self._apply_func(pred_dict, *args)

    def predict_proba(self, *args, **kwargs):
        if not self._fitted:
            raise AttributeError('Please fit the Vset object before calling the predict_proba method.')
        pred_dict = {}
        for k, v in self.out.items():
            if hasattr(v, 'predict_proba'):
                pred_dict[k] = v.predict_proba
        return self._apply_func(pred_dict, *args)

    def evaluate(self, *args, **kwargs):
        '''Combines dicts before calling _apply_func
        '''
        return self._apply_func(None, *args)

    def __call__(self, *args, n_out: int = None, keys: list = [], **kwargs):
        '''
        '''
        if n_out is None:
            n_out = len(args)
        out = sep_dicts(self._apply_func(None, *args), n_out=n_out, keys=keys)
        return out

    def __getitem__(self, i):
        '''Accesses ith item in the module set
        '''
        return self.modules[i]

    def __contains__(self, key):
        '''Returns true if modules is a dict and key is one of its keys
        '''
        if isinstance(self.modules, dict):
            return key in self.modules.keys()
        return False

    def keys(self):
        if isinstance(self.modules, dict):
            return self.modules.keys()
        return {}.keys()

    def __len__(self):
        return len(self.modules)

    def __str__(self):
        return 'Vset(' + self.name + ')'

    def __create_subkey(self, value):
        return Subkey(value, self.name, self._output_matching)


def _apply_func_cached(out_dict: dict, is_async: bool, lazy: bool, *args):
    '''
    Params
    ------
    *args: List[Dict]: takes multiple dicts and combines them into one.
            Then runs modules on each item in combined dict.
    out_dict: the dictionary to pass to the matching function. If None, defaults to self.modules.

    Returns
    -------
    results: dict
        with items being determined by functions in module set.
        Functions and input dictionaries are currently matched using  matching = 'cartesian' format.
            e.g. inputs:    module = {LR : logistic}, data = {train_1 : [X1,y1], train2 : [X2,y2]}
                 out:    out_dict = {(train_1, LR)  : fitted logistic, (train_2, LR) :  fitted logistic}.
        Currently matching = 'subset' is not used...
    '''
    async_args = []
    for in_dict in args:
        if not isinstance(in_dict, dict):
            raise Exception('Need to run init_args before calling module_set!')
        if is_async:
            remote_dict = {}
            # send data to the remote object store
            for k, v in in_dict.items():
                if k != PREV_KEY:
                    remote_dict[k] = ray.put(v)
                else:
                    remote_dict[k] = v
            async_args.append(remote_dict)
    if is_async:
        args = async_args
    data_dict = combine_dicts(*args)
    out_dict = apply_modules(out_dict, data_dict, lazy)

    if is_async:
        out_keys = list(out_dict.keys())
        out_vals = ray.get(list(out_dict.values()))
        out_dict = dict(zip(out_keys, out_vals))

    return data_dict, out_dict
