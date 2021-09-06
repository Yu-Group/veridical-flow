import numpy as np
from sklearn.dummy import DummyClassifier

from vflow import ModuleSet, init_args  # must install vflow first (pip install vflow)
from vflow.convert import combine_three_subset_dicts, combine_two_dicts, extend_dicts, full_combine_two_dicts


class TestDictCombine():

    def setup(self):
        pass

    def test_extend_dict(self):
        arg_names = ['x_train', 'y_train', 'classifiers']
        vals_tuple = ('x_train', 'y_train', ['log_reg'])
        x_train, y_train, classifiers = init_args(vals_tuple, names=arg_names)
        combined_dict = extend_dicts(x_train, y_train, classifiers)
        for arg in arg_names:
            assert arg in combined_dict.keys(), 'dict should contain ' + str(arg) + ' as key'
        for val in vals_tuple:
            assert val in combined_dict.values(), 'dict should contain ' + str(val) + ' as val'

    def test_full_combine_dicts(self):
        x_train, y_train = init_args(('x_train', 'y_train'), names=['x_train', 'y_train'])
        merged_dict = extend_dicts(x_train, y_train)
        non_merged = {('classifier 1', 'rf'): lambda x: x, ('classifier 2', 'dt'): lambda x: x}
        combined_dict = full_combine_two_dicts(merged_dict, non_merged)
        k1 = (('classifier 1', 'rf'), 'x_train', 'y_train')
        k2 = (('classifier 2', 'dt'), 'x_train', 'y_train')
        for k in [k1, k2]:
            assert k in combined_dict.keys(), 'dict should contain ' + str(k) + ' as key'

    def test_combine_three_dicts(self):
        x_train, y_train = init_args(('x_train', 'y_train'), names=['x_train', 'y_train'])
        classifiers = {('classifier 1', 'rf'): 0, ('classifier 2', 'dt'): 1}
        combined_dict = combine_three_subset_dicts(x_train, y_train, classifiers)
        k1 = (('classifier 1', 'rf'), 'x_train', 'y_train')
        k2 = (('classifier 2', 'dt'), 'x_train', 'y_train')
        for k in [k1, k2]:
            assert k in combined_dict.keys(), 'dict should contain ' + str(k) + ' as key'
        v1 = [0, 'x_train', 'y_train']
        v2 = [1, 'x_train', 'y_train']
        for v in [v1, v2]:
            assert v in combined_dict.values(), 'dict should contain ' + str(v) + ' as value'

    def test_combine_subset_dicts(self):
        x_train = x_test = np.array([-1, 1, 1, 1])
        y_train = y_test = np.array([0, 1, 1, 1])
        x_train, x_test, y_train, y_test = init_args((x_train, x_test, y_train, y_test),
                                                     names=['x_train', 'x_test', 'y_train', 'y_test'])
        dummy_subsample_fns = [lambda x, y: (x, y) for i in range(3)]
        dummy_samsample_set = ModuleSet(name='subsample', modules=dummy_subsample_fns)
        x_trains, y_trains = dummy_samsample_set(x_train, y_train)
        x_tests, y_tests = dummy_samsample_set(x_test, y_test)
        dummy_modeling_set = ModuleSet(name='modeling', modules=[DummyClassifier()])
        dummy_modeling_set.fit(x_trains, y_trains)
        dummy_preds = dummy_modeling_set.predict(x_tests)
        # y_tests keys of form ('x_test', 'y_test', 'subsample_0') should match with dummy_preds
        combined_dict = combine_two_dicts(dummy_preds, y_tests)
        k1 = (('x_train', 'y_train', 'subsample_0', 'modeling_0'), ('x_test', 'y_test', 'subsample_0'))
        assert k1 in combined_dict, 'dict should contain ' + str(k1) + ' as key'
