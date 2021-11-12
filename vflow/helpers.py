'''User-facing helper functions included at import vflow
'''

from vflow.vset import Vset
from vflow.vfunc import Vfunc

from itertools import product
from functools import partial

def build_vset(name: str, obj, param_dict: dict = {}, *args,
               is_async: bool = False, output_matching: bool = False,
               lazy: bool = False, cache_dir: str = None,
               tracking_dir: str = None, **kwargs) -> Vset:
    '''Builds a Vset by currying callable obj with all combinations of parameters in param_dict.

    Params
    -------
    name: str
        a name for the output Vset
    obj: callable
        a callable to use as the base for Vfuncs in the output Vset
    param_dict: dict[str, list]
        keys are obj kwarg names and values in the dict are lists of params to try
    *args
        additional fixed arguments to pass to obj
    is_async: bool (optional)
        if True, modules are computed asynchronously
    output_matching: bool (optional)
        if True, then output keys from Vset will be matched when used
        in other Vsets
    cache_dir: str (optional)
        if provided, do caching and use cache_dir as the data store for
        joblib.Memory
    tracking_dir: str (optional)
        if provided, use the mlflow.tracking api to log outputs as metrics
        with params determined by input keys
    **kwargs
        additional fixed keyword arguments to pass to obj

    Returns
    -------
    new_vset : Vset
    '''
    assert callable(obj), 'obj must be callable'

    vfuncs = []

    # TODO: better way to check this?
    instantiate = isinstance(obj, type)

    param_names = list(param_dict.keys())
    param_lists = list(param_dict.values())
    kwargs_tuples = product(*param_lists)
    for tup in kwargs_tuples:
        kwargs_dict = {}
        for param_name, param_val in zip(param_names, tup):
            kwargs_dict[param_name] = param_val
        # add additional fixed kwargs
        for k, v in kwargs.items():
            kwargs_dict[k] = v
        if instantiate:
            # instantiate obj
            vfuncs.append(Vfunc(module=obj(*args, **kwargs_dict)))
        else:
            # use partial to wrap obj
            vfuncs.append(Vfunc(module=partial(obj, *args, **kwargs_dict)))

    return Vset(name, vfuncs, is_async=is_async,
                output_matching=output_matching, lazy=lazy,
                cache_dir=cache_dir, tracking_dir=tracking_dir)
