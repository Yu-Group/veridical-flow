import pcs


class TestBasic():
    def setup(self):
        self.pipeline = pcs.Pipeline()

    def test_import(self):
        assert self.pipeline.test is False
