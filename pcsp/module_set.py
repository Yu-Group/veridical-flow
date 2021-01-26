'''Set of perturbations to be used in pipeline
'''


class ModuleSet:
    def __init__(self, name: str = '', modules: list = []):
        self.name = name
        self.modules = modules

    def fit(self, *args, **kwargs):
        for mod in self.modules:
            mod.fit(*args, **kwargs)

    def transform(self, *args, **kwargs):
        results = []
        for mod in self.modules:
            result = mod.transform(*args, **kwargs)
            results.append(result)
        return results

    def predict(self, *args, **kwargs):
        results = []
        for mod in self.modules:
            result = mod.predict(*args, **kwargs)
            results.append(result)
        return results

    def predict_proba(self, *args, **kwargs):
        results = []
        for mod in self.modules:
            result = mod.predict(*args, **kwargs)
            results.append(result)
        return results

    def __call__(self, *args, **kwargs):
        results = []
        for mod in self.modules:
            result = mod(*args, **kwargs)
            results.append(result)
        return results

    def __getitem__(self, i):
        '''Accesses ith item in the module set
        '''
        return self.modules[i]

    def __len__(self):
        return len(self.modules)

    def __str__(self):
        return self.name + ': ' + ','.join([str(mod) for mod in self.modules])
