import pytest

import vflow
from vflow.utils import to_tuple, to_list


class TestBasic:
    def setup(self):
        self.pipeline = vflow.PCSPipeline()
        self.module_set = vflow.Vset(name='s', modules={})
        self.module = vflow.Vfunc()

    def test_class_initializations(self):
        assert self.pipeline.steps is not None
        assert self.module_set.modules is not None
        assert self.module is not None

    def test_iteration(self):
        """Tests that iterating over pipeline is same as iterating over its steps
        """
        self.pipeline.steps = [0, 1, 2]
        assert self.pipeline.steps[0] == 0
        assert self.pipeline[0] == 0, 'accessing pipeline steps'
        for i, x in enumerate(self.pipeline):
            assert x == i, 'iterating over pipeline steps'
        assert self.pipeline[1:] == [1, 2], 'slicing pipeline'

    def test_list_packing(self):
        """Test that packing / unpacking lists works appropriately
        """
        start = [[0, 10], [1, 11], [2, 12]]
        X, y = to_tuple(start)
        packed = to_list((X, y))
        assert start == packed, 'unpacking/packing works'
    
    def test_to_list(self):
        assert to_list((['x1', 'x2', 'x3'], ['y1', 'y2', 'y3'])) == [['x1', 'y1'], ['x2', 'y2'], ['x3', 'y3']]
        assert to_list((['x1'], ['y1'])) == [['x1', 'y1']]
        assert to_list((['x1', 'x2', 'x3'],)) == [['x1'], ['x2'], ['x3']]
        assert to_list(('x1', )) == [['x1']]
        assert to_list(('x1', 'y1')) == [['x1', 'y1']]
        assert to_list(('x1', 'x2', 'x3', 'y1', 'y2', 'y3')) == [['x1', 'y1'], ['x2', 'y2'], ['x3', 'y3']]
        with pytest.raises(ValueError):
            to_list(('x1', 'x2', 'x3', 'y1', 'y2'))
