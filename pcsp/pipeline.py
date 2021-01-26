'''Class that stores the entire pipeline of steps in a data-science workflow
'''


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
