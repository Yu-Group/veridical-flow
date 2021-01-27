'''Set of modules to be parallelized over in a pipeline.
Function arguments are each a list
'''


def s(x):
    if type(x) in [list, tuple]:
        return len(x)
    else:
        return x.shape


class ModuleSet:
    def __init__(self, name: str = '', modules: list = []):
        self.name = name
        self.modules = modules

    def fit(self, *args, **kwargs):
        '''todo: support kwargs
        '''
        funcs = [mod.fit for mod in self.modules]
        return self.apply_func(funcs, *args, **kwargs)

    def apply_func(self, funcs, *args, **kwargs):
        '''

        Params
        ------
        *args: Tuple[List]
            should be unpacked list
            e.g. ([x1, x2, x3], [y1, y2, y3])

        Returns
        -------
        results: Tuple[List]
            e.g. ([x1, x2, x3], [y1, y2, y3])
        '''
        # print('intro args', s(args))
        # if len(args) == 1:
        #     args = args[0]
        # args = args[0]
        # print('args_tuple', s(args), s(args[0]), 'kwargs', kwargs)  # , args[0][0].shape)
        # args should look like ([x1, x2, x3], [y1, y2, y3])
        args_list = to_list(args)
        # args_list should look like [[x1, y1], [x2, y2], [x3, y3]]
        # print('args_list', s(args_list), s(args_list[0]))  # , args_list[0][0].shape, 'kwargs', kwargs)
        results = []
        for arg in args_list:
            # print('arg', s(arg), s(arg[0]))  # , s(arg[0]), s(arg[1]))  # , arg[0].shape)  # , arg[1].shape)
            for func in funcs:
                result = func(*arg)
                results.append(result)
        # print(results)
        # print('result', s(result), s(result[0]), s(result[1]), s(result[0][0]))
        # print('len(results)', len(results), len(results[0]), s(results[0][0]), s(results[0][1]))
        return to_tuple(results)

    def transform(self, *args, **kwargs):
        results = []
        for mod in self.modules:
            result = mod.transform(*args, **kwargs)
            results.append(result)
        return results

    def predict(self, *args, **kwargs):
        funcs = [mod.predict for mod in self.modules]
        return self.apply_func(funcs, *args, **kwargs)

    def predict_proba(self, *args, **kwargs):
        funcs = [mod.predict for mod in self.modules]
        return self.apply_func(funcs, *args, **kwargs)

    def repeat(self, x):
        '''

        Parameters
        ----------
        x: list to be repeated

        Returns
        -------
        List repeated number of times as self.modules
        '''

        return x * len(self.modules)

    def __call__(self, *args, **kwargs):
        return self.apply_func(self.modules, *args, **kwargs)

    def __getitem__(self, i):
        '''Accesses ith item in the module set
        '''
        return self.modules[i]

    def __len__(self):
        return len(self.modules)

    def __str__(self):
        return self.name + ': ' + ','.join([str(mod) for mod in self.modules])


def to_tuple(lists: list):
    '''Convert from lists to unpacked  tuple
    Ex. [[x1, y1], [x2, y2], [x3, y3]] -> ([x1, x2, x3], [y1, y2, y3])
    Ex. [[x1, y1]] -> ([x1], [y1])
    Ex. [m1, m2, m3] -> [m1, m2, m3]
    Allows us to write X, y = ([x1, x2, x3], [y1, y2, y3])
    '''
    n_mods = len(lists)
    if n_mods <= 1:
        return lists
    if not type(lists[0]) == list:
        return lists
    n_tup = len(lists[0])
    tup = [[] for _ in range(n_tup)]
    for i in range(n_mods):
        for j in range(n_tup):
            tup[j].append(lists[i][j])
    return tuple(tup)


def to_list(tup: tuple):
    '''Convert from tuple to packed list
    Ex. ([x1, x2, x3], [y1, y2, y3]) -> [[x1, y1], [x2, y2], [x3, y3]]
    Ex. ([x1], [y1]) -> [[x1, y1]]
    Ex. ([x1, x2, x3]) -> [[x1], [x2], [x3]]
    Allows us to call function with arguments in a loop
    '''
    n_tup = len(tup)
    if n_tup == 0:
        return tup
    elif n_tup == 1:
        return [[x] for x in tup[0]]
    n_mods = len(tup[0])
    lists_packed = [[] for _ in range(n_mods)]
    for i in range(n_mods):
        for j in range(n_tup):
            lists_packed[i].append(tup[j][i])
    return lists_packed
