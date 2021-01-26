import pcsp


class TestBasic():
    def setup(self):
        self.pipeline = pcsp.PCSPipeline()
        self.perturbation_set = pcsp.PerturbationSet()
        self.perturbation = pcsp.Perturbation()

    def test_class_initializations(self):
        assert self.pipeline.steps is not None
        assert self.perturbation_set.perturbations is not None
        assert self.perturbation is not None

    def test_iteration(self):
        '''Tests that iterating over pipeline is same as iterating over its steps
        '''
        self.pipeline.steps = [0, 1, 2]
        assert self.pipeline.steps[0] == 0
        assert self.pipeline[0] == 0, 'accessing pipeline steps'
        for i, x in enumerate(self.pipeline):
            assert x == i, 'iterating over pipeline steps'
        assert self.pipeline[1:] == [1, 2], 'slicing pipeline'
