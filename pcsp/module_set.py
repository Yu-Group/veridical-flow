'''Set of modules to be parallelized over in a pipeline.
Function arguments are each a list
'''
PREV_KEY = '__prev__'
from pcsp.convert import *
import joblib

class ModuleSet:
    def __init__(self, name: str, modules, module_keys: list=None, out : dict = {}):
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
        module_outs: dict 
            saved outs corresponding to each module after passing in data. 
        '''
        self.name = name
        self._fitted = False
        self.out = out
        if type(modules) is dict:
            self.modules = modules
        elif type(modules) is list:
            if module_keys is not None:
                assert type(module_keys) is list, 'modules passed as list but module_names is not a list'
                assert len(modules) == len(module_keys), 'modules list and module_names list differ'
            else:
                module_keys = [f'{name}_{i}' for i in range(len(modules))]
            self.modules = dict(zip(module_keys, modules))

    def apply_func(self, *args, matching='cartesian', order='typical', use_out=False, replace=False,**kwargs):
        '''

        Params
        ------
        *args: List[Dict]: takes multiple dicts and combines them into one.
                Then runs modules on each item in combined dict. 
        use_out
            Should we store the out into .out? If we store it in .out, don't call sep_dicts before returning
        
        Returns
        -------
        results: dict
            with items being determined by functions in module set.
            Functions and input dictionaries are currently matched using  matching = 'cartesian' format.
                e.g. inputs:    module = {LR : logistic}, data = {train_1 : [X1,y1], train2 : [X2,y2]}
                     out:    out_dict = {(train_1, LR)  : fitted logistic, (train_2, LR) :  fitted logistic}.
            Currently matching = 'subset' is not used...
        '''
        # print('apply func!')
        
        for ele in args:
            if not isinstance(ele, dict):
                raise Exception('Need to run init_args before calling module_set!')
        

        data_dict = combine_dicts(*args)
#             print(data_dict.keys())
        if matching == 'cartesian':
            if use_out:
                out_dict = cartesian_dict(data_dict, self.out, order=order)
            else:
                out_dict = cartesian_dict(data_dict, self.modules, order=order)
        elif matching == 'subset':
            out_dict = subset_dict(data_dict, self.modules, order=order)
        else:
            out_dict = {}
        #for key,val in out_dict.items(): 
        #    print(key)
        #for key,val in self.out.items():
        #    print("sef keys:" + str(key))
        #self.out.update(out_dict)
        # add PREV_KEY
#             print('prev', str(data_dict[PREV_KEY]))
        self.__prev__ = data_dict[PREV_KEY]
        out_dict[PREV_KEY] = self
        if replace == False: 
            self.out.update(out_dict)
        else:
            self.out = out_dict
        # store out_dict in modules
        #self.modules = out_dict
        #for key,val in out_dict.items(): 
        #    print(key)
        #    self.out[key] = val
#         print(out_dict)
        if use_out:
            return out_dict
        else:
            dicts_separated = sep_dicts(out_dict)
#         print('\n\nsep\n', dicts_separated)
            return dicts_separated
        # out_dict = cartesian_dict(combine_dicts(*args))
        # data_dict = append_dict(*args)
        # out_dict = cartesian_dict(*args,self.modules)


    def fit(self, *args, **kwargs):
        '''todo: support kwargs
        '''
        # funcs = [mod.fit for mod in self.modules.items()]
        if self._fitted:
            return self
        # atm, module is not necessarily a Module object
        for k1, v1 in self.modules.items():
            if hasattr(v1, 'fit'):
                self.modules[k1] = v1.fit
        self.apply_func(*args, matching='cartesian', order='typical', use_out = False, replace = False,**kwargs)
        self._fitted = True
        return self
        
    def transform(self, *args, **kwargs):
        results = []
        for mod in self.modules:
            result = mod.transform(*args, **kwargs)
            results.append(result)
        return results

    def predict(self, *args, **kwargs):
        #for k1, v1 in self.modules.items():
        #    self.modules[k1] = v1.predict
        for k1,v1 in self.out.items():
            self.out[k1] = v1.predict
        return self.apply_func(*args, matching='cartesian', order='backwards', use_out=True, replace=True, **kwargs)

    def predict_proba(self, *args, **kwargs):
        for k1, v1 in self.modules.items():
            self.modules[k1] = v1.predict
        return self.apply_func(*args, matching='cartesian', order='backwards', **kwargs)

    def evaluate(self, *args, **kwargs):
        '''Combines dicts before calling apply_func
        '''
        validation_dict = combine_subset_dicts(*args, order='typical')
        self.fit(validation_dict, **kwargs)
        return self.out
        # for k1, v1 in self.modules.items():
        #    return self.apply_func(*args,matching = 'subset',order = 'typical',**kwargs)

    def __call__(self, *args, **kwargs):
        return self.apply_func(*args, matching='cartesian', order='typical', use_out=False, replace=False, **kwargs)

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
