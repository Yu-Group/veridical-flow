'''A perturbation that can be used as a step in a pipeline
'''
from abc import abstractmethod

import ray

class Module:
    '''Module is basically a function along with a name attribute.
    It may support a "fit" function, but may also just have a "transform" function.
    If none of these is supported, it need only be a function
    '''

    def __init__(self, name: str='', module=lambda x: x):
        assert hasattr(module, 'fit') or callable(module), \
            'module must be an object with a fit method or a callable'
        self.name = name
        self.module = module

    def fit(self, *args, **kwargs):
        '''This function fits params for this module
        '''
        if hasattr(self.module, 'fit'):
            return self.module.fit(*args, **kwargs)
        else:
            return self.module(*args, **kwargs)

    @abstractmethod
    def transform(self, *args, **kwargs):
        '''This function transforms its input in some way
        '''
        pass

    def __call__(self, *args, **kwargs):
        '''This should decide what to call
        '''
        return self.fit(*args, **kwargs)

@ray.remote
def _remote_fun(module, *args, **kwargs):
    return module(*args, **kwargs)

class AsyncModule:
    '''An asynchronous version of the Module class.
    '''
    def __init__(self, name: str='', module=lambda x: x, *args, **kwargs):
        self.name = name
        if isinstance(module, Module):
            self.module = module.module
        else:
            assert hasattr(module, 'fit') or callable(module),\
                'module must be an object with a fit method or a callable'
            self.module = module

    def fit(self, *args, **kwargs):
        '''This function fits params for this module
        '''
        if hasattr(self.module, 'fit'):
            return _remote_fun.remote(self.module.fit, *args, **kwargs)
        else:
            return _remote_fun.remote(self.module, *args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.fit(*args, **kwargs)
