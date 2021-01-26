import pcsp
from pcsp.module_set import unpack_to_tuple, pack_to_list


class TestBasic():
    def setup(self):
        self.pipeline = pcsp.PCSPipeline()
        self.module_set = pcsp.ModuleSet()
        self.module = pcsp.Module()

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

    def test_packing(self):
        '''Test that packing / unpacking work appropriately
        '''
        start = [[0, 10], [1, 11], [2, 12]]
        X, y = unpack_to_tuple(start)
        packed = pack_to_list((X, y))
        assert start == packed, 'unpacking/packing works'
