'''Useful functions for converting between different types (dicts, lists, tuples, etc.)
'''

def s(x):
    '''Gets shape of a list/tuple/ndarray
    '''
    if type(x) in [list, tuple]:
        return len(x)
    else:
        return x.shape
    
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


def sep_dicts(output: dict):
    ''' converts dictionary with value being saved as a tuple/list into multiple dictionaries. 
    Assumes every value has same length of tuple  
    '''
    n_dict = len(output)
    if n_dict == 0:  # empty dict -- return empty dict
        return {}
    else:
        try:
            n_tup = len(tuple(output.items())[0][1])  # first item in list
            sep_dict = [dict() for x in range(n_tup)]
            for key, value in output.items():
                for i in range(n_tup):
                    sep_dict[i][key] = value[i]
            return sep_dict
        except:
            return output


def combine_dicts(*args):
    '''Combines dicts into one dict with values now being a list of items from input dicts. 
    Assume all args are dicts,keys are the same and are the same length'''
    n_tup = len(args)
    if n_tup == 0:
        return {}
    elif n_tup == 1:
        return args[0]
    else:
        combined_dict = {}
        for key, val in args[0].items():
            items = []
            for i in range(n_tup):
                items.append(args[i][key])
            combined_dict[key] = items
        return combined_dict


def combine_subset_dicts(*args, order='typical'):
    '''Combines dicts into one dict with values now being a list of items from input dicts. only combines dicts with key from one      dictionary is subset of other dictionaries last value in key. Assumes that keys are tuples. 
    '''
    n_tup = len(args)
    print(n_tup)
    if n_tup == 0:
        return {}
    elif n_tup == 1:
        return args[0]
    else:
        combined_dict = {}
        for key1, val1 in args[0].items():
            for key2, val2 in args[1].items():
                if (set(key1).issubset(key2[-1]) and order == 'typical'):
                    combined_dict[key2] = [val1, val2]
                elif (set(key1).issubset(key2[-1]) and order != 'typical'):
                    combined_dict[key2] = [val2, val1]
                else:
                    combined_dict = combined_dict
        return combined_dict


def create_dict(*args):
    ''' converts *args which is in format Tuple(list) to dict. For example if args = ([x1, x2, x3], [y1, y2, y3]), then 
     dict = {item1: [x1,y1], item2: [x2,y2], item3: [x3,y3]}. Assume each element of tuple has same len.
    '''
    output_dict = {}
    args_list = to_list(args)
    for (i, ele) in enumerate(args_list):
        key = "data_" + str(i)
        output_dict[key] = args_list[i]
    return output_dict


def cartesian_dict(data, modules, order='typical'):
    '''returns cartessian product of two dictionaries. order refers in which order new keys are added. e.g. order = typical means new key will be (k1,k2). else new key will be (k2,k1). order != typical is used for predict function.'''
    cart = {}
    for k1, v1 in data.items():
        for k2, v2 in modules.items():
            if not isinstance(k1, tuple):
                if isinstance(v1, tuple):
                    if (order == 'typical'):
                        cart.update({(k1, k2): v2(*v1)})
                    else:
                        cart.update({(*k2, k1): v2(*v1)})  # *k2
                elif isinstance(v1, list):
                    if (order == 'typical'):
                        cart.update({(k1, k2): v2(*v1)})
                    else:
                        cart.update({(*k2, k1): v2(*v1)})  # *k2
                else:
                    if (order == 'typical'):
                        cart.update({(k1, k2): v2(v1)})
                    else:
                        cart.update({(*k2, k1): v2(v1)})  # *k2
            else:
                if isinstance(v1, tuple):
                    if (order == 'typical'):
                        cart.update({(*k1, k2): v2(*v1)})  # *k1
                    else:
                        cart.update({(k2, k1): v2(*v1)})
                elif isinstance(v1, list):
                    if (order == 'typical'):
                        cart.update({(*k1, k2): v2(*v1)})  # *k1
                    else:
                        cart.update({(k2, k1): v2(*v1)})
                else:
                    if (order == 'typical'):
                        cart.update({(*k1, k2): v2(v1)})  # *k1
                    else:
                        cart.update({(k2, k1): v2(v1)})

    return cart


'''
def cartesian_dict_tuple(data,modules,order = 'typical'):
    cart = {}
    for k1, v1 in data.items():
        for k2, v2 in modules.items():
            if not isinstance(k1, tuple):
                if isinstance(v1,tuple):
                    if(order == 'typical'):
                        cart.update({((k1,), (k2,)): v2(*v1)})
                    else:
                        cart.update({((k2,),(k1,)): v2(*v1)}) #*k2
                elif isinstance(v1,list):
                    if(order == 'typical'):
                        cart.update({((k1,), (k2,)): v2(*v1)})
                    else:
                        cart.update({((k2,), (k1,)): v2(*v1)}) #*k2
                else:
                    if(order == 'typical'):
                        cart.update({((k1,), (k2,)): v2(v1)})
                    else:
                         cart.update({((k2,), (k1,)): v2(v1)}) #*k2
            else:
                if isinstance(v1,tuple):
                    if(order == 'typical'):
                        cart.update({((k1,), (k2,)): v2(*v1)}) #*k1
                    else:
                        cart.update({((k2,), (k1,)): v2(*v1)})
                elif isinstance(v1,list):
                    if(order == 'typical'):
                        cart.update({((k1,), (k2)): v2(*v1)}) #*k1
                    else:
                        cart.update({((k2,), (k1,)): v2(*v1)})
                else:
                    if(order == 'typical'):
                        cart.update({((k1,) ,(k2,)): v2(v1)}) #*k1
                    else:
                        cart.update({((k2,), (k1,)): v2(v1)})
                
    return cart
'''


def subset_dict(data, modules, order='typical'):
    cart = {}
    for k1, v1 in data.items():
        for k2, v2 in modules.items():
            if (set(k1).issubset(k2)):
                if not isinstance(k1, tuple):
                    if isinstance(v1, tuple):
                        if (order == 'typical'):
                            cart.update({(k1, k2): v2(*v1)})
                        else:
                            cart.update({(*k2, k1): v2(*v1)})  # *k2
                    elif isinstance(v1, list):
                        if (order == 'typical'):
                            cart.update({(k1, k2): v2(*v1)})
                        else:
                            cart.update({(*k2, k1): v2(*v1)})  # *k2
                    else:
                        if (order == 'typical'):
                            cart.update({(k1, k2): v2(v1)})
                        else:
                            cart.update({(*k2, k1): v2(v1)})  # *k2
                else:
                    if isinstance(v1, tuple):
                        if (order == 'typical'):
                            cart.update({(*k1, k2): v2(*v1)})  # *k1
                        else:
                            cart.update({(k2, k1): v2(*v1)})
                    elif isinstance(v1, list):
                        if (order == 'typical'):
                            cart.update({(*k1, k2): v2(*v1)})  # *k1
                        else:
                            cart.update({(k2, k1): v2(*v1)})
                    else:
                        if (order == 'typical'):
                            cart.update({(*k1, k2): v2(v1)})  # *k1
                        else:
                            cart.update({(k2, k1): v2(v1)})
    return cart
