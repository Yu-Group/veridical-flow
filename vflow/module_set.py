'''Set of modules to be parallelized over in a pipeline.
Function arguments are each a list
'''
PREV_KEY = '__prev__'

from vflow.convert import *
from vflow.module import Module, AsyncModule

from copy import deepcopy

import ray
import numpy as np

class ModuleSet:
    def __init__(self, name: str, modules, module_keys: list=None, is_async: bool=False):
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
        '''
        self.name = name
        self._fitted = False
        self.out = None # outputs
        self._async = is_async
        # check if any of the modules are AsyncModules
        # if so, we'll make then all AsyncModules later on
        if not self._async and np.any([isinstance(mod, AsyncModule) for mod in modules]):
            self._async = True
        if type(modules) is dict:
            self.modules = modules
        elif type(modules) is list:
            if module_keys is not None:
                assert type(module_keys) is list, 'modules passed as list but module_names is not a list'
                assert len(modules) == len(module_keys), 'modules list and module_names list differ'
            else:
                module_keys = [f'{name}_{i}' for i in range(len(modules))]
            self.modules = dict(zip(module_keys, modules))
        # if needed, wrap the modules in the Module or AsyncModule class
        for k, v in self.modules.items():
            if self._async:
                if not isinstance(v, AsyncModule):
                    self.modules[k] = AsyncModule(k, v)
            elif not isinstance(v, Module):
                self.modules[k] = Module(k, v)

    def apply_func(self, *args, out_dict=None, matching='cartesian', order='typical', **kwargs):
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
        if out_dict is None:
            out_dict = self.modules

        # deepcopy args to avoid mutating them
        args = deepcopy(args)

        for ele in args:
            if not isinstance(ele, dict):
                raise Exception('Need to run init_args before calling module_set!')
            if self._async:
                # send data to the remote object store
                for k, v in ele.items():
                    if k != PREV_KEY:
                        ele[k] = ray.put(v)


        # combine two dicts via cartesian if either has length 1 (ignoring prev)
        # does subset matching if both have more than length 1 
        data_dict = combine_two_dicts(*args, order=order)
        if matching == 'cartesian':
            if 'match_on' in kwargs:
                out_dict = cartesian_dict(data_dict, out_dict, order=order, match_on=kwargs['match_on'])
            else:
                out_dict = cartesian_dict(data_dict, out_dict, order=order)
        elif matching == 'subset':
            out_dict = subset_dict(data_dict, out_dict, order=order)
        else:
            out_dict = {}

        if self._async:
            out_keys = list(out_dict.keys())
            out_vals = ray.get(list(out_dict.values()))
            out_dict = dict(zip(out_keys, out_vals))

        self.__prev__ = data_dict[PREV_KEY]
        out_dict[PREV_KEY] = self

        return out_dict


    def fit(self, *args, **kwargs):
        '''
        '''
        if self._fitted:
            return self
        out_dict = {}
        for k, v in self.modules.items():
            out_dict[k] = v.fit
        self.out = self.apply_func(*args, out_dict=out_dict, matching='cartesian', order='typical', **kwargs)
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

    def predict(self, *args, match_on=None, **kwargs):
        if not self._fitted:
            raise AttributeError('Please fit the ModuleSet object before calling the predict method.')
        pred_dict = {}
        for k, v in self.out.items():
            if hasattr(v, 'predict'):
                pred_dict[k] = v.predict
        return self.apply_func(*args, out_dict=pred_dict, matching='cartesian', order='backwards', match_on=match_on, **kwargs)

    def predict_proba(self, *args, **kwargs):
        if not self._fitted:
            raise AttributeError('Please fit the ModuleSet object before calling the predict_proba method.')
        pred_dict = {}
        for k, v in self.out.items():
            if hasattr(v, 'predict_proba'):
                pred_dict[k] = v.predict
        return self.apply_func(*args, out_dict=pred_dict, matching='cartesian', order='backwards', **kwargs)

    def evaluate(self, *args, **kwargs):
        '''Combines dicts before calling apply_func
        '''
        return self.apply_func(*args, matching='cartesian', order='typical', **kwargs)

    def __call__(self, *args, **kwargs):
        '''Save into self.out, or append to self.out
        '''
        out = self.apply_func(*args, out_dict=self.modules,
                              matching='cartesian', order='typical', **kwargs)
        if self.out is None:
            self.out = [out]
        else:
            self.out.append(out)
        return sep_dicts(out)

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
        return 'ModuleSet(' + self.name  + ')'
