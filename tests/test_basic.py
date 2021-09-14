import vflow
from vflow.convert import to_tuple, to_list


class TestBasic():
    def setup(self):
        self.pipeline = vflow.PCSPipeline()
        self.module_set = vflow.ModuleSet(name='s', modules={})
        self.module = vflow.Module()

    def test_class_initializations(self):
        assert self.pipeline.steps is not None
        assert self.module_set.modules is not None
        assert self.module is not None

    def test_iteration(self):
        '''Tests that iterating over pipeline is same as iterating over its steps
        '''
        self.pipeline.steps = [0, 1, 2]
        assert self.pipeline.steps[0] == 0
        assert self.pipeline[0] == 0, 'accessing pipeline steps'
        for i, x in enumerate(self.pipeline):
            assert x == i, 'iterating over pipeline steps'
        assert self.pipeline[1:] == [1, 2], 'slicing pipeline'

    def test_list_packing(self):
        '''Test that packing / unpacking lists works appropriately
        '''
        start = [[0, 10], [1, 11], [2, 12]]
        X, y = to_tuple(start)
        packed = to_list((X, y))
        assert start == packed, 'unpacking/packing works'
