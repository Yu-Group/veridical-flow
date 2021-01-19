import pcs


class TestBasic():
    def setup(self):
        self.pipeline = pcs.Pipeline()
        self.perturbation_set = pcs.PerturbationSet()
        self.perturbation = pcs.Perturbation()

    def test_import(self):
        assert self.pipeline.preprocess is False
        assert self.perturbation_set.perturbations is not None
        assert self.perturbation is not None
