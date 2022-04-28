import importlib
import inspect

def get_imported_module(module_path):
    """ return class/function module """
    module = importlib.import_module(module_path)
    if inspect.isclass(module):
        return module()
    else:
        return module
