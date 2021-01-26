'''A perturbation that can be used as a step in a pipeline
'''
from abc import abstractmethod


class Module:
    '''Module is basically a function along with a name attribute.
    It may support a "fit" function, but may also just have a "transform" function.
    If none of these is supported, it need only be a function
    '''

    def __init__(self):
        self.name = ''

    @abstractmethod
    def fit(self, *args, **kwargs):
        '''This function fits params for this module
        '''
        pass

    @abstractmethod
    def transform(self, *args, **kwargs):
        '''This function transforms its input in some way
        '''
        pass

    def __call__(self, *args, **kwargs):
        '''This should decide what to call
        '''
        if 'fit' in dir(self):
            self.fit(*args, **kwargs)
        return self.transform(*args, **kwargs)
