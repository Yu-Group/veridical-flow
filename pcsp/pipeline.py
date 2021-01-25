'''Class that stores the entire pipeline of steps in a data-science workflow
'''
from sklearn.pipeline import Pipeline


class PCSPipeline(Pipeline):
    def __init__(self):
        self.steps = []
        self.preprocess = False

    def run(self):
        '''Runs the pipeline

        '''
        pass

    def __getitem__(self, i):
        '''Accesses ith step of pipeline
        '''
        return self.steps[i]

    def __len__(self):
        return len(self.steps)
