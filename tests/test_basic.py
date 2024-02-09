import pytest

import vflow
from vflow.utils import to_list, to_tuple


class TestBasic:
    def setup_method(self):
        self.pipeline = vflow.PCSPipeline()
        self.vfunc_set = vflow.Vset(name="s", vfuncs={})
        self.vfunc = vflow.Vfunc()

    def test_class_initializations(self):
        assert self.pipeline.steps is not None
        assert self.vfunc_set.vfuncs is not None
        assert self.vfunc is not None

    def test_iteration(self):
        """Tests that iterating over pipeline is same as iterating over its steps"""
        self.pipeline.steps = [0, 1, 2]
        assert self.pipeline.steps[0] == 0
        assert self.pipeline[0] == 0, "accessing pipeline steps"
        for i, x in enumerate(self.pipeline):
            assert x == i, "iterating over pipeline steps"
        assert self.pipeline[1:] == [1, 2], "slicing pipeline"

    def test_list_packing(self):
        """Test that packing / unpacking lists works appropriately"""
        start = [[0, 10], [1, 11], [2, 12]]
        X, y = to_tuple(start)
        packed = to_list((X, y))
        assert start == packed, "unpacking/packing works"

    def test_to_list(self):
        assert to_list((["x1", "x2", "x3"], ["y1", "y2", "y3"])) == [
            ["x1", "y1"],
            ["x2", "y2"],
            ["x3", "y3"],
        ]
        assert to_list((["x1"], ["y1"])) == [["x1", "y1"]]
        assert to_list((["x1", "x2", "x3"],)) == [["x1"], ["x2"], ["x3"]]
        assert to_list(("x1",)) == [["x1"]]
        assert to_list(("x1", "y1")) == [["x1", "y1"]]
        assert to_list(("x1", "x2", "x3", "y1", "y2", "y3")) == [
            ["x1", "y1"],
            ["x2", "y2"],
            ["x3", "y3"],
        ]
        with pytest.raises(ValueError):
            to_list(("x1", "x2", "x3", "y1", "y2"))

    def test_build_graph(self):
        v0 = vflow.Vset("v0", [lambda x: x + 1], ["add1"])
        v1 = vflow.Vset("v1", [lambda x: 2 * x], ["mult2"])
        v2 = vflow.Vset("v2", [lambda x: x % 3], ["mod3"])

        x = vflow.init_args([1.5], ["x"])[0]
        x0 = v0.fit_transform(x)
        x1 = v1.fit_transform(x0)
        x2 = v2.fit_transform(x1)

        graph = vflow.build_graph(x2)
        assert graph.is_directed()
        assert graph.size() == 4  # edges
        assert graph.order() == 5  # nodes: init + 3 Vsets + End
        in_degrees = dict(graph.in_degree).values()
        assert max(in_degrees) == 1
        assert sum(in_degrees) == 4
        edges = list(graph.edges)
        assert ("init", v0) in edges
        assert (v0, v1) in edges
        assert (v1, v2) in edges
        assert (v2, "End") in edges
