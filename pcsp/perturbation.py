'''A perturbation that can be used as a step in a pipeline
'''
from abc import abstractmethod

class Perturbation:
    def __init__(self):
        self.name = ''

    @abstractmethod
    def transform(self, *args, **kwargs):
        '''This function transforms its input in some way
        '''
        pass