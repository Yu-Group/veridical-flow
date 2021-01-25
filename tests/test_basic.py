import pcsp


class TestBasic():
    def setup(self):
        self.pipeline = pcsp.PCSPipeline()
        self.perturbation_set = pcsp.PerturbationSet()
        self.perturbation = pcsp.Perturbation()

    def test_import(self):
        assert self.pipeline.preprocess is False
        assert self.perturbation_set.perturbations is not None
        assert self.perturbation is not None
