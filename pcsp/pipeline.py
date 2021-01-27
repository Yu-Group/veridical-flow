'''Class that stores the entire pipeline of steps in a data-science workflow
'''
import itertools

import pandas as pd


class PCSPipeline:
    def __init__(self, steps: list = []):
        self.steps = steps
        self.cache = []

    def run(self, *args, **kwargs):
        '''Runs the pipeline
        '''
        for i, step in enumerate(self.steps):
            try:
                step_name = step.name
            except:
                step_name = f'Step {i}'
            outputs = step(*args, **kwargs)
            self.cache.append((step_name, outputs))

    def __getitem__(self, i):
        '''Accesses ith step of pipeline
        '''
        return self.steps[i]

    def __len__(self):
        return len(self.steps)

    def generate_names(self, as_pandas=True):
        name_lists = []
        if as_pandas:
            for step in self.steps:
                name_lists.append([f'{i}_{str(mod)[:8]}'
                                   for i, mod in enumerate(step)])
            indexes = list(itertools.product(*name_lists))
            return pd.DataFrame(indexes, columns=[step.name for step in self.steps])
        else:
            for step in self.steps:
                name_lists.append([f'{step.name}_{i}_{str(mod)[:8]}'
                                   for i, mod in enumerate(step)])
            return list(itertools.product(*name_lists))
