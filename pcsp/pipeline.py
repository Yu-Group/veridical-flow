'''Class that stores the entire pipeline of steps in a data-science workflow
'''


class PCSPipeline:
    def __init__(self, steps: list=[]):
        self.steps = steps

    def fit(self, *args, **kwargs):
        '''Runs the pipeline
        '''
        for step in self.steps:
            step.fit(*args, **kwargs)

    def __getitem__(self, i):
        '''Accesses ith step of pipeline
        '''
        return self.steps[i]

    def __len__(self):
        return len(self.steps)
