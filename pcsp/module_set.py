'''Set of modules to be parallelized over in a pipeline.
Function arguments are each a list
'''
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
        if type(modules) is dict:
            self.modules = modules
        elif type(modules) is list:
            if module_keys is not None:
                assert type(module_keys) is list, 'modules passed as list but module_names is not a list'
                assert len(modules) == len(module_keys), 'modules list and module_names list differ'
            else:
                module_keys = [f'{name}_{i}' for i in range(len(modules))]
            self.modules = dict(zip(module_keys, modules))
        

    def fit(self, *args, **kwargs):
        '''todo: support kwargs
        '''
        # funcs = [mod.fit for mod in self.modules.items()]
        for k1, v1 in self.modules.items():
            self.modules[k1] = v1.fit
        return self.apply_func(*args, matching='cartesian', order='typical', **kwargs)

    def apply_func(self, *args, matching='cartesian', order='typical', **kwargs):
        '''

        Params
        ------
        *args: Two types currently allowed: 
            - Tuple[List] e.g. ([X1 ,X2],[y1, y2]).
                apply_func will then combine *args into a dictionary of format: 
                {data_0 : [X1, y1], data_1: [X2, y2]} and run modules on each item in dictionary
            - Dictionaries: takes multiple dicts and combines them into one.
                Then runs modules on each item in combined dict. 
        
        Returns
        -------
        results: dict
            with items being determined by functions in module set.
            Functions and input dictionaries are currently matched using  matching = 'cartesian' format.
                e.g. inputs:    module = {LR : logistic}, data = {train_1 : [X1,y1], train2 : [X2,y2]}
                     output:    output_dict = {(train_1, LR)  : fitted logistic, (train_2, LR) :  fitted logistic}.
            Currently matching = 'subset' is not used...
        '''
        dict_present = False
        for ele in args:
            if isinstance(ele, dict):
                dict_present = True  # Checking if dict is present

        # if dictionary is present, create dictionary with default data_0 key
        if not dict_present:
            data_dict = create_dict(*args)
            if matching == 'cartesian':
                output_dict = cartesian_dict(data_dict, self.modules, order=order)
            elif matching == 'subset':
                output_dict = subset_dict(data_dict, self.modules, order=order)
            else:
                output_dict = {}
            self.modules = output_dict
            return sep_dicts(output_dict)
        
        # if dictionary is not present, combine dicts based on keys
        else:
            data_dict = combine_dicts(*args)
            if (matching == 'cartesian'):
                output_dict = cartesian_dict(data_dict, self.modules, order=order)
            elif (matching == 'subset'):
                output_dict = subset_dict(data_dict, self.modules, order=order)
            else:
                output_dict = {}
            self.modules = output_dict
            # output_dict = cartesian_dict(combine_dicts(*args))
            return output_dict
            # data_dict = append_dict(*args)
            # output_dict = cartesian_dict(*args,self.modules)


    def transform(self, *args, **kwargs):
        results = []
        for mod in self.modules:
            result = mod.transform(*args, **kwargs)
            results.append(result)
        return results

    def predict(self, *args, **kwargs):
        for k1, v1 in self.modules.items():
            self.modules[k1] = v1.predict
        return self.apply_func(*args, matching='cartesian', order='backwards', **kwargs)

    def predict_proba(self, *args, **kwargs):
        for k1, v1 in self.modules.items():
            self.modules[k1] = v1.predict
        return self.apply_func(*args, matching='cartesian', order='backwards', **kwargs)

    def evaluate(self, *args, **kwargs):
        validation_dict = combine_subset_dicts(*args, order='typical')
        return self.apply_func(validation_dict, matching='cartesian', order='typical', **kwargs)
        # for k1, v1 in self.modules.items():
        #    return self.apply_func(*args,matching = 'subset',order = 'typical',**kwargs)

    def __call__(self, *args, **kwargs):
        return self.apply_func(*args, matching='cartesian', order='typical', **kwargs)

    def __getitem__(self, i):
        '''Accesses ith item in the module set
        '''
        return self.modules[i]

    def __len__(self):
        return len(self.modules)

    def __str__(self):
        return self.name + ': ' + ','.join([str(mod) for mod in self.modules])