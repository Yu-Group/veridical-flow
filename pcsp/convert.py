'''Useful functions for converting between different types (dicts, lists, tuples, etc.)
'''
import enum
from turtle import left, right
from pcsp.module_set import PREV_KEY, MATCH_KEY
from pcsp.smart_subkey import SmartSubkey
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

# def combine_keys(left_key, right_key, left_match_ids, right_match_ids):
    left_num_matches = len(left_match_ids)
    right_num_matches = len(right_match_ids)

    if left_num_matches < right_num_matches:
        match_key = left_key
        match_ids = left_match_ids
        compare_key = right_key
        compare_ids = right_match_ids
    else:
        match_key = right_key
        match_ids = right_match_ids
        compare_key = left_key
        compare_ids = left_match_ids
    if len(match_ids) > 0:
        matched_subkeys = []
        for idx, subkey in enumerate(match_key):
            if idx in match_ids and isinstance(subkey, SmartSubkey):
                for id in compare_ids:
                    c_subkey = compare_key[id]
                    if isinstance(c_subkey, SmartSubkey) and subkey == c_subkey:
                        matched_subkeys.append(subkey)
                        break
        if len(matched_subkeys) > 0:
            unmatched_subkeys = [
                compare_key[idx] for idx in compare_ids
                if compare_key[idx] not in matched_subkeys
            ]
            filtered_key = tuple([subkey for subkey in right_key if subkey not in matched_subkeys])
            combined_key = left_key + filtered_key
            new_match_ids = [
                i for i in range(len(combined_key))
                if combined_key[i] in matched_subkeys
                or combined_key[i] in unmatched_subkeys
            ]
            return combined_key, new_match_ids 
        else:
            return (), []
    else:
        # no matching needed, just concatenate key tuples
        if left_num_matches > 0:
            new_match_ids = left_match_ids
        elif right_num_matches > 0:
            new_match_ids = [idx + len(left_key) for idx in right_match_ids]
        else:
            new_match_ids = []
        return left_key + right_key, new_match_ids

def combine_keys(left_key, right_key):
    if len(left_key) < len(right_key):
        match_key = left_key
        compare_key = right_key
    else:
        match_key = right_key
        compare_key = left_key
    match_ids = []
    for idx, subkey in enumerate(match_key):
        if isinstance(subkey, SmartSubkey):
            match_ids.append(idx)
    if len(match_ids) > 0:
        matched_subkeys = []
        for idx in match_ids:
            subkey = match_key[idx]
            for c_subkey in compare_key:
                if isinstance(c_subkey, SmartSubkey):
                    if subkey.origin == c_subkey.origin:
                        if subkey.subkey == c_subkey.subkey:
                            matched_subkeys.append(subkey)
                            break
                        else:
                            return ()
        if len(matched_subkeys) > 0:
            filtered_key = tuple([subkey for subkey in right_key if subkey not in matched_subkeys])
            combined_key = left_key + filtered_key
            return combined_key
        else:
            return left_key + right_key
    else:
        return left_key + right_key

# def combine_keys(left_key, right_key, left_match_ids, right_match_ids):
    '''Combines the keys into a single key, possibly matching on the keys in
    positions determined by match_ids. Whichever key has fewer matching
    requirements is the key we match on, and if one of the keys has no
    match_ids then just concatenates the keys. If both match_ids are non-empty
    and no match is found, returns an empty tuple. Returns the new key and new
    match_ids for the combined key.
    '''
    left_num_matches = len(left_match_ids)
    right_num_matches = len(right_match_ids)

    # match on the arg that requires the smallest number of matching subkeys
    if left_num_matches < right_num_matches:
        match_key = left_key
        match_ids = left_match_ids
        compare_key = right_key
        compare_match_ids = right_match_ids
    else:
        match_key = right_key
        match_ids = right_match_ids
        compare_key = left_key
        compare_match_ids = left_match_ids

    num_matches = len(match_ids)
    if num_matches > 0:
        # matching
        matched_subkeys = []
        for idx, subkey in enumerate(match_key):
            if idx in match_ids and subkey in compare_key:
                matched_subkeys.append(subkey)
        if len(matched_subkeys) == num_matches:
            # positive match: combine keys, making sure to preserve subkey order
            # find subkeys in compare_key that went unmatched
            unmatched_subkeys = [
                compare_key[idx] for idx in compare_match_ids
                if compare_key[idx] not in matched_subkeys
            ]
            # the right key is always filtered
            filtered_key = tuple([
                subkey for subkey in right_key if subkey not in matched_subkeys
            ])
            combined_key = left_key + filtered_key
            new_match_ids = [
                i for i in range(len(combined_key))
                if combined_key[i] in matched_subkeys
                or combined_key[i] in unmatched_subkeys
            ]
            return combined_key, new_match_ids
        else:
            # no match
            return (), []
    else:
        # no matching needed, just concatenate key tuples
        if left_num_matches > 0:
            new_match_ids = left_match_ids
        elif right_num_matches > 0:
            new_match_ids = [idx + len(left_key) for idx in right_match_ids]
        else:
            new_match_ids = []
        return left_key + right_key, new_match_ids


def combine_dicts(*args: dict, base_case=True):
    '''Combines any number of dictionaries into a single dictionary. Dictionaries
    are combined left to right, matching on the subkeys of the arg that has
    fewer matching requirements.
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

        left_num_matches = right_num_matches = 0
        left_match_ids = []
        right_match_ids = []

        if MATCH_KEY in args[0]:
            left_match_ids = args[0][MATCH_KEY]

        if MATCH_KEY in args[1]:
            right_match_ids = args[1][MATCH_KEY]

        match_ids = []

        for k0 in args[0]:
            for k1 in args[1]:

                if k0 in KEYS or k1 in KEYS:
                    continue

                # NOTE: having combine_keys() return new_match_ids is a
                # temporary solution... TODO: implement a ModuleSetKey class to
                # wrap the key tuples and keep the key's match_ids
                # combined_key, new_match_ids = combine_keys(
                #     k0, k1, left_match_ids, right_match_ids
                # )
                combined_key = combine_keys(k0, k1)

                # NOTE: this is pretty sloppy and should be fixed by having
                # match_ids on a per-key basis rather than for an entire
                # dictionary
                # if len(new_match_ids) > len(match_ids):
                #     match_ids = new_match_ids

                if len(combined_key) > 0:
                    if base_case:
                        combined_dict[combined_key] = (args[0][k0], args[1][k1])
                    else:
                        combined_dict[combined_key] = args[0][k0] + (args[1][k1], )

        prev_tup = ()
        for i in range(2):
            if PREV_KEY in args[i]:
                prev_tup += args[i][PREV_KEY]
        combined_dict[PREV_KEY] = prev_tup
        combined_dict[MATCH_KEY] = match_ids
        return combined_dict
    else:
        # combine the first two dicts and call recursively with remaining args
        return combine_dicts(combine_dicts(args[0], args[1]), *args[2:], base_case=False)


def apply_modules(modules: dict, data_dict: dict):
    out_dict = {}
    num_matches = 0

    mod_num_matches = data_num_matches = 0
    mod_match_ids = []
    data_match_ids = []

    if MATCH_KEY in modules:
        mod_match_ids = modules[MATCH_KEY]
        mod_num_matches = len(mod_match_ids)

    if MATCH_KEY in data_dict:
        data_match_ids = data_dict[MATCH_KEY]
        data_num_matches = len(data_match_ids)

    match_ids = []

    for mod_k in modules:
        for data_k in data_dict:
            if mod_k in KEYS or data_k in KEYS:
                continue
            # combined_key, new_match_ids = combine_keys(
            #     data_k, mod_k, data_match_ids, mod_match_ids
            # )
            combined_key = combine_keys(data_k, mod_k)

            # if len(new_match_ids) > len(match_ids):
            #     match_ids = new_match_ids

            if len(combined_key) > 0:
                out_dict[combined_key] = deepcopy(modules[mod_k])(*data_dict[data_k])

    out_dict[MATCH_KEY] = match_ids
    return out_dict
