'''Useful functions for converting between different types (dicts, lists, tuples, etc.)
'''
from pcsp.module_set import PREV_KEY, MATCH_KEY
from copy import deepcopy

KEYS = [PREV_KEY, MATCH_KEY]

def init_args(args_tuple: tuple, names=None):
    ''' converts tuple of arguments to a list of dicts
    Params
    ------
    names: optional, list-like
        gives names for each of the arguments in the tuple
    '''
    if names is None:
        names = ['start'] * len(args_tuple)
    else:
        assert len(names) == len(args_tuple), 'names should be same length as args_tuple'

    output_dicts = []
    for (i, ele) in enumerate(args_tuple):
        output_dicts.append({
            (names[i], ): args_tuple[i],
            PREV_KEY: ('init', ),
        })
    return output_dicts


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
    Ex. (x1) -> [[x1]]
    Ex. (x1, y1) -> [[x1, y1]]
    Ex. (x1, x2, x3, y1, y2, y3) -> [[x1, y1], [x2, y2], [x3, y3]]
    Ex. (x1, x2, x3, y1, y2) -> Error
    Allows us to call function with arguments in a loop
    '''
    n_tup = len(tup)
    if n_tup == 0:
        return []
    elif not isinstance(tup[0], list):
        # the first element is data
        if n_tup == 1:
            return list(tup)
        if n_tup % 2 != 0:
            raise ValueError('Don\'t know how to handle uneven number of args '
                             'without a list. Please wrap your args in a list.')
        # assume first half of args is input and second half is outcome
        return [list(el) for el in zip(tup[:(n_tup//2)], tup[(n_tup//2):])]
    elif n_tup == 1:
        return [[x] for x in tup[0]]
    n_mods = len(tup[0])
    lists_packed = [[] for _ in range(n_mods)]
    for i in range(n_mods):
        for j in range(n_tup):
            lists_packed[i].append(tup[j][i])
    return lists_packed


def sep_dicts(d: dict):
    '''converts dictionary with value being saved as a tuple/list into multiple dictionaries
    Assumes every value has same length of tuple

    Params
    ------
    d: {k1: (x1, y1), k2: (x2, y2), ...,  '__prev__': p}

    Returns
    -------
    sep_dicts: [{k1: x1, k2: x2, ..., '__prev__': p}, {k1: y1, k2: y2, '__prev__': p}]
    '''

    # empty dict -- return empty dict
    n_dicts = len(d)
    if n_dicts == 0:
        return {}
    else:
        # try separating dict into multiple dicts
        val_list_len = len(tuple(d.values())[0])  # first item in list
        sep_dicts = [dict() for x in range(val_list_len)]
        for key, value in d.items():
            if not key in KEYS:
                for i in range(val_list_len):
                    # assumes the correct sub-key for item i is in the i-th position
                    new_key = (key[i], ) + key[val_list_len:]
                    sep_dicts[i][new_key] = value[i]

        # add back prev
        prev = d[PREV_KEY]
        match_ids = d[MATCH_KEY] if MATCH_KEY in d else []
        for i, idx in enumerate(match_ids):
            match_ids[i] -= len(sep_dicts) - 1
        for i in range(len(sep_dicts)):
            sep_dicts[i][PREV_KEY] = prev
            sep_dicts[i][MATCH_KEY] = match_ids
        return sep_dicts


def combine_dicts(*args: dict, base_case=True):
    '''Combines any number of dictionaries into a single dictionary. If MATCH_KEY
    is found in dict d, then the number of matching sub-keys in each tuple key
    must equal d[MATCH_KEY]. Dictionaries are combined left to right, with the
    rightmost arg determining the matching. As such, this method is not
    commutative when d[MATCH_KEY] differs for input dicts and d[MATCH_KEY] != 0
    for at least one input.
    '''
    n_args = len(args)
    combined_dict = {}
    if n_args == 0:
        return combined_dict
    elif n_args == 1:
        for k in args[0]:
            # wrap the dict values in tuples; this is helpful so that when we
            # pass the values to a module fun in we just can use * expansion
            if k not in KEYS:
                combined_dict[k] = (args[0][k], )
            else:
                combined_dict[k] = args[0][k]
        return combined_dict
    elif n_args == 2:

        num_matches = 0
        if MATCH_KEY in args[1]:
            match_ids = args[1][MATCH_KEY]
            num_matches = len(match_ids)

        for k0 in args[0]:
            for k1 in args[1]:
                if k0 in KEYS or k1 in KEYS:
                    continue
                if num_matches > 0:
                    # matching
                    subkey_matches = 0
                    for idx, subkey in enumerate(k1):
                        if idx in match_ids and subkey in k0:
                            subkey_matches += 1
                    if subkey_matches == num_matches:
                        # positive match
                        # make sure to preserve sub-key order
                        # keeps the last occurrence of a non-unique sub-key
                        filtered_k1 = tuple([
                            k1[i] for i in range(len(k1)) if i not in match_ids
                        ])
                        combined_key = k0 + filtered_k1
                        if base_case:
                            combined_dict[combined_key] = (args[0][k0], args[1][k1])
                        else:
                            combined_dict[combined_key] = args[0][k0] + (args[1][k1], )
                else:
                    combined_key = k0 + k1
                    # no matching, just combine everything
                    if base_case:
                        combined_dict[combined_key] = (args[0][k0], args[1][k1])
                    else:
                        combined_dict[combined_key] = args[0][k0] + (args[1][k1], )

        prev_tup = ()
        for i in range(2):
            if PREV_KEY in args[i]:
                prev_tup += args[i][PREV_KEY]
        combined_dict[PREV_KEY] = prev_tup
        combined_dict[MATCH_KEY] = []
        return combined_dict
    else:
        # combine the first two dicts and call recursively with remaining args
        return combine_dicts(combine_dicts(args[0], args[1]), *args[2:], base_case=False)


def combine_two_dicts(*args, order='typical'):
    '''assume len args is at most 2 for now.
    If either dictionary has only 1 key(ignoring prev), then we do a cartesian match.
    If both dictionaries has more than 1 key(ignoring prev), then we do a subset dictionary match via combine_two_subset_dicts
    TODO: Allow more than length 2 and Make keys look better.
    '''
    n_args = len(args)
    if n_args == 0:
        return {}
    elif n_args == 1:
        return args[0]
    elif n_args == 2:
        combined_dict = {}
        if(len(args[0].keys()) <= 2 or len(args[1].keys()) <= 2):
            for key0,val0 in args[0].items():
                for key1,val1 in args[1].items():
                    if key0 == PREV_KEY or key1 == PREV_KEY:
                        continue
                    else:
                        combined_dict[(key0,key1)] = [val0,val1]
                        #combined_dict[key0  (key1,)] = [val0,val1]
            prev_list = []
            for i in range(n_args):
                if PREV_KEY in args[i]:
                    prev_list.append(args[i][PREV_KEY])
            combined_dict[PREV_KEY] = prev_list
            #print('combine', prev_list)
            return combined_dict
        else:
            return combine_two_subset_dicts(*args, order=order)
    else:
        return combine_three_subset_dicts(*args, order=order)


def combine_two_subset_dicts(*args, order='typical'):
    '''Combines dicts into one dict.
    Values are now a list of items from input dicts.
    only combines dicts where (last value in) key from one dictionary 
    is subset of other dictionaries key.
    Assumes that keys are tuples. 
    '''
    n_args = len(args)
    if n_args == 0:
        return {}
    elif n_args == 1:
        return args[0]
    else:
        combined_dict = {}
        for key1, val1 in args[0].items():
            for key2, val2 in args[1].items():
                if key1 == PREV_KEY or key2 == PREV_KEY:
                    continue
                else:
                    if key2[-1] in set(key1) or key2 in set(key1):
                        if order == 'typical':
                            combined_dict[key1] = [val1, val2]
                        elif order != 'typical':
                            combined_dict[key2] = [val2, val1]
                    elif key1[-1] in set(key2) or key1 in set(key2):
                        if order == 'typical':
                            combined_dict[key2] = [val1, val2]
                        elif order != 'typical':
                            combined_dict[key1] = [val2, val1]
                    else:
                        combined_dict = combined_dict
        # add prev_keys from all previous dicts as a list
        prev_list = []
        for i in range(n_args):
            if PREV_KEY in args[i]:
                prev_list.append(args[i][PREV_KEY])
        combined_dict[PREV_KEY] = prev_list
        return combined_dict


def combine_three_subset_dicts(*args, order='typical'):
    if order == 'typical':
        sorted_args = sorted(list(args), key=lambda x: len(x.items()))
    else:
        sorted_args = sorted(list(args), key=lambda x: len(x.items()), reverse=True)
    merged_dict = extend_dicts(*sorted_args[:2])
    combined_dict = full_combine_two_dicts(merged_dict, sorted_args[-1])
    prev_list = []
    for i in range(len(args)):
        if PREV_KEY in args[i]:
            prev_list.append(args[i][PREV_KEY])
    combined_dict[PREV_KEY] = prev_list
    return combined_dict


def full_combine_two_dicts(merged: dict, non_merged: dict):
    '''Cartesian matching of dictionary non_merged
    with all keys of dictionary merged treated as a unit
    TODO: add keyword arg to reverse tuple ordering?
    '''
    combined_dict = {}
    for key, val in non_merged.items():
        if key != PREV_KEY:
            combined_dict[(key, *merged.keys())] = [val, *merged.values()]
    return combined_dict


def extend_dicts(*args):
    '''Extends dicts into one dict
    Assumes all keys are unique (except PREV_KEY)
    '''
    combined_dict = {}
    for arg in args:
        for key, val in arg.items():
            if key != PREV_KEY:
                combined_dict[key] = val
    return combined_dict


def combine_subset_dicts(*args, order='typical'):
    '''DEPRECATED
    Combines dicts into one dict.
    Values are now a list of items from input dicts.
    only combines dicts with key from one dictionary is subset of other dictionaries last value in key.
    Assumes that keys are tuples.
    '''
    n_args = len(args)
#     print('subset', n_args)
    if n_args == 0:
        return {}
    elif n_args == 1:
        return args[0]
    else:
        combined_dict = {}
        for key1, val1 in args[0].items():
            for key2, val2 in args[1].items():
                if key1 == PREV_KEY or key2 == PREV_KEY:
                    continue
                else:
                    if set(key1).issubset(key2[-1]) and order == 'typical':
                        combined_dict[key2] = [val1, val2]
                    elif set(key1).issubset(key2[-1]) and order != 'typical':
                        combined_dict[key2] = [val2, val1]
                    else:
                        combined_dict = combined_dict

        # add prev_keys from all previous dicts as a list
        prev_list = []
        for i in range(n_args):
            if PREV_KEY in args[i]:
                prev_list.append(args[i][PREV_KEY])
        combined_dict[PREV_KEY] = prev_list
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


def apply_modules(modules: dict, data_dict: dict):
    out_dict = {}
    num_matches = 0

    if MATCH_KEY in data_dict:
        match_ids = data_dict[MATCH_KEY]
        num_matches += len(match_ids)

    for mod_k in modules:
        for data_k in data_dict:
            if mod_k in KEYS or data_k in KEYS:
                continue
            combined_key = data_k + mod_k
            if num_matches > 0:
                # matching
                subkey_matches = 0
                for idx, subkey in enumerate(data_k):
                    if idx in match_ids and subkey in mod_k:
                        subkey_matches += 1
                if subkey_matches == num_matches:
                    # positive match
                    # make sure to preserve sub-key order
                    filtered_data_k = tuple([
                        data_k[i] for i in range(len(data_k)) if i not in match_ids
                    ])
                    combined_key = filtered_data_k + mod_k
                    out_dict[combined_key] = deepcopy(modules[mod_k])(*data_dict[data_k])
            else:
                # no matching, just combine via cartesian
                combined_key = data_k + mod_k
                out_dict[combined_key] = deepcopy(modules[mod_k])(*data_dict[data_k])

    return out_dict


def cartesian_dict(data, modules, order: str='typical', match_on=None):
    '''returns cartesian product of two dictionaries
    Params
    ------
    order: str
        refers in which order new keys are added.
        e.g. order = typical means new key will be (k1,k2).
        else new key will be (k2,k1). order != typical is used for predict function.
    '''
    cart = {}
    for k1, v1 in data.items():
        for k2, v2 in modules.items():
            if k1 == PREV_KEY or k2 == PREV_KEY:
                continue
#             print(k1, k2)
            try:
                # deepcopy the method so that original modules are not modified
                # e.g., when v2 is a sklearn model .fit method
                v2 = deepcopy(v2)
                if match_on is None or k1[match_on] in set(k2):
                    if not isinstance(k1, tuple):
                        if isinstance(v1, tuple):
                            if order == 'typical':
                                cart.update({(k1, k2): v2(*v1)})
                            else:
                                cart.update({(*k2, k1): v2(*v1)})  # *k2
                        elif isinstance(v1, list):
                            if order == 'typical':
                                cart.update({(k1, k2): v2(*v1)})
                            else:
                                cart.update({(*k2, k1): v2(*v1)})  # *k2
                        else:
                            if order == 'typical':
                                cart.update({(k1, k2): v2(v1)})
                            else:
                                cart.update({(*k2, k1): v2(v1)})  # *k2
                    else:
                        if isinstance(v1, tuple):
                            if order == 'typical':
                                cart.update({(*k1, k2): v2(*v1)})  # *k1
                            else:
                                cart.update({(k2, k1): v2(*v1)})
                        elif isinstance(v1, list):
                            if order == 'typical':
                                cart.update({(*k1, k2): v2(*v1)})  # *k1
                            else:
                                cart.update({(k2, k1): v2(*v1)})
                        else:
                            if order == 'typical':
                                cart.update({(*k1, k2): v2(v1)})  # *k1
                            else:
                                cart.update({(k2, k1): v2(v1)})
            except Exception as e:
                print(e)
    return cart


def subset_dict(data, modules, order='typical'):
    cart = {}
    for k1, v1 in data.items():
        for k2, v2 in modules.items():
            if set(k1).issubset(k2):
                if not isinstance(k1, tuple):
                    if isinstance(v1, tuple):
                        if order == 'typical':
                            cart.update({(k1, k2): v2(*v1)})
                        else:
                            cart.update({(*k2, k1): v2(*v1)})  # *k2
                    elif isinstance(v1, list):
                        if order == 'typical':
                            cart.update({(k1, k2): v2(*v1)})
                        else:
                            cart.update({(*k2, k1): v2(*v1)})  # *k2
                    else:
                        if order == 'typical':
                            cart.update({(k1, k2): v2(v1)})
                        else:
                            cart.update({(*k2, k1): v2(v1)})  # *k2
                else:
                    if isinstance(v1, tuple):
                        if order == 'typical':
                            cart.update({(*k1, k2): v2(*v1)})  # *k1
                        else:
                            cart.update({(k2, k1): v2(*v1)})
                    elif isinstance(v1, list):
                        if order == 'typical':
                            cart.update({(*k1, k2): v2(*v1)})  # *k1
                        else:
                            cart.update({(k2, k1): v2(*v1)})
                    else:
                        if order == 'typical':
                            cart.update({(*k1, k2): v2(v1)})  # *k1
                        else:
                            cart.update({(k2, k1): v2(v1)})
    return cart
