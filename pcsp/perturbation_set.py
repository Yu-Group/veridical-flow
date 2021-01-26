'''Set of perturbations to be used in pipeline
'''


class PerturbationSet:
    def __init__(self):
        self.perturbations = []

    def fit(self, *args, **kwargs):
        for perturbation in self.perturbations:
            perturbation.fit(*args, **kwargs)
