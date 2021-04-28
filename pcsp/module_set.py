'''Set of modules to be parallelized over in a pipeline.
Function arguments are each a list
'''
PREV_KEY = '__prev__'
from pcsp.convert import *

class ModuleSet:
    def __init__(self, name: str, modules, module_keys: list=None):
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
        '''
        self.name = name
        self._fitted = False
        self.out = None # outputs
        if type(modules) is dict:
            self.modules = modules
        elif type(modules) is list:
            if module_keys is not None:
                assert type(module_keys) is list, 'modules passed as list but module_names is not a list'
                assert len(modules) == len(module_keys), 'modules list and module_names list differ'
            else:
                module_keys = [f'{name}_{i}' for i in range(len(modules))]
            self.modules = dict(zip(module_keys, modules))

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
        for ele in args:
            if not isinstance(ele, dict):
                raise Exception('Need to run init_args before calling module_set!')
        
        if out_dict is None:
            out_dict = self.modules
            
        # combine two dicts via cartesian if either has length 1 (ignoring prev)
        # does subset matching if both have more than length 1 
        data_dict = combine_two_dicts(*args) 
        if matching == 'cartesian':
            out_dict = cartesian_dict(data_dict, out_dict, order=order)
        elif matching == 'subset':
            out_dict = subset_dict(data_dict, out_dict, order=order)
        else:
            out_dict = {}
        self.__prev__ = data_dict[PREV_KEY]
        out_dict[PREV_KEY] = self

        return out_dict


    def fit(self, *args, **kwargs):
        '''
        '''
        if self._fitted:
            return self
        # atm, module is not necessarily a Module object
        out_dict = {}
        for k, v in self.modules.items():
            if hasattr(v, 'fit'):
                out_dict[k] = v.fit
            else:
                out_dict[k] = v
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

    def predict(self, *args, **kwargs):
        if not self._fitted:
            raise AttributeError('Please fit the ModuleSet object before calling the predict method.')
        pred_dict = {}
        for k, v in self.out.items():
            if hasattr(v, 'predict'):
                pred_dict[k] = v.predict
        return self.apply_func(*args, out_dict=pred_dict, matching='cartesian', order='backwards', **kwargs)

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
        return self.apply_func(*args,matching = 'cartesian',order = 'typical',**kwargs)

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
