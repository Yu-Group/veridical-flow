'''Set of modules to be parallelized over in a pipeline.
Function arguments are each a list
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
        # print('args', len(args), len(args[0]))
        results = []
        for mod_num, mod in enumerate(self.modules):
            result = mod(*args, **kwargs)
            results.append(result)
            # results[mod_num] = result
        return unpack_to_tuple(results)

    def __getitem__(self, i):
        '''Accesses ith item in the module set
        '''
        return self.modules[i]

    def __len__(self):
        return len(self.modules)

    def __str__(self):
        return self.name + ': ' + ','.join([str(mod) for mod in self.modules])


def unpack_to_tuple(lists_packed: list):
    '''Convert from lists to unpacked  tuple
    Ex. [[x1, y1], [x2, y2], [x3, y3]] -> ([x1, x2, x3], [y1, y2, y3])
    Allows us to write X, y = ([x1, x2, x3], [y1, y2, y3])
    '''
    n_packed = len(lists_packed)
    if n_packed == 0:
        return []
    n_unpacked = len(lists_packed[0])
    lists_unpacked = [[] for _ in range(n_unpacked)]
    for i in range(n_packed):
        for j in range(n_unpacked):
            lists_unpacked[j].append(lists_packed[i][j])
    return tuple(lists_unpacked)


def pack_to_list(lists_tuple: tuple):
    '''Convert from tuple to packed list
    Ex. ([x1, x2, x3], [y1, y2, y3]) -> [[x1, y1], [x2, y2], [x3, y3]]
    Allows us to call function with arguments in a loop
    '''
    n_unpacked = len(lists_tuple)
    if n_unpacked == 0:
        return []
    n_packed = len(lists_tuple[0])
    lists_packed = [[] for _ in range(n_packed)]
    for i in range(n_packed):
        for j in range(n_unpacked):
            lists_packed[i].append(lists_tuple[j][i])
    return lists_packed
