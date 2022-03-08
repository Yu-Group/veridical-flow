from numpy.testing import assert_equal

from vflow.helpers import *

class TestHelpers:

    def test_build_vset(self):

        def my_func(param1: str, param2: str, param3: str='a'):
            return (param1, param2, param3)

        def my_func2(param1: str, param2: str, param3: str='b'):
            return (param1 + '1', param2 + '2', param3)

        param_dict1 = { 'param1': ['hello', 'foo'], 'param2': ['world', 'bar'] }
        param_dict2 = { 'param1': ['hello'], 'param2': ['world', 'there']}

        # my_func without param_dict
        vset = build_vset("vset", my_func, param1='hello', param2='world', param3='b')
        assert len(vset) == 1, \
            'build_vset with my_func fails'
        d_key = [key[0] for key in list(vset.vfuncs.keys())][0]
        assert d_key.value == 'vset_0', \
            'build_vset with my_func fails'
        d_keyword = [val.vfunc.keywords for val in list(vset.vfuncs.values())][0]
        assert d_keyword == {'param1': 'hello', 'param2': 'world', 'param3': 'b'}, \
            'build_vset with my_func fails'

        # my_func without param_dict, reps
        vset = build_vset("vset", my_func, reps=2, param1='hello', param2='world', param3='b')
        assert len(vset) == 2, \
            'build_vset with my_func + reps fails'
        d_keys = [key[0].value[0] for key in list(vset.vfuncs.keys())]
        assert d_keys[0] == 'rep=0', \
            'build_vset with my_func + reps fails'
        assert d_keys[1] == 'rep=1', \
            'build_vset with my_func + reps fails'
        d_keywords = [val.vfunc.keywords for val in list(vset.vfuncs.values())]
        assert d_keywords[0] == {'param1': 'hello', 'param2': 'world', 'param3': 'b'}, \
            'build_vset with my_func + reps fails'
        assert d_keywords[1] == {'param1': 'hello', 'param2': 'world', 'param3': 'b'}, \
            'build_vset with my_func + reps fails'

        # my_func with param_dict1
        vset = build_vset("vset", my_func, param_dict1, param3='b')
        assert len(vset) == 4, \
            'build_vset with my_func + param_dict1 fails'
        d_keys = [key[0] for key in list(vset.vfuncs.keys())]
        assert d_keys[0].value == ('func=my_func', 'param1=hello', 'param2=world'), \
            'build_vset with my_func + param_dict1 fails'
        assert d_keys[1].value == ('func=my_func', 'param1=hello', 'param2=bar'), \
            'build_vset with my_func + param_dict1 fails'
        assert d_keys[2].value == ('func=my_func', 'param1=foo', 'param2=world'), \
            'build_vset with my_func + param_dict1 fails'
        assert d_keys[3].value == ('func=my_func', 'param1=foo', 'param2=bar'), \
            'build_vset with my_func + param_dict1 fails'
        d_keywords = [val.vfunc.keywords for val in list(vset.vfuncs.values())]
        assert d_keywords[0] == {'param1': 'hello', 'param2': 'world', 'param3': 'b'}, \
            'build_vset with my_func + param_dict1 fails'
        assert d_keywords[1] == {'param1': 'hello', 'param2': 'bar', 'param3': 'b'}, \
            'build_vset with my_func + param_dict1 fails'
        assert d_keywords[2] == {'param1': 'foo', 'param2': 'world', 'param3': 'b'}, \
            'build_vset with my_func + param_dict1 fails'
        assert d_keywords[3] == {'param1': 'foo', 'param2': 'bar', 'param3': 'b'}, \
            'build_vset with my_func + param_dict1 fails'

        # my_func with param_dict2, reps
        vset = build_vset("vset", my_func, param_dict2, reps=2, lazy=True, param3='b')
        assert vset._lazy, \
            'build_vset with my_func + param_dict2 + reps fails'
        assert len(vset) == 4, \
            'build_vset with my_func + param_dict2 + reps fails'
        d_keys = [key[0] for key in list(vset.vfuncs.keys())]
        assert d_keys[0].value == ('rep=0', 'func=my_func', 'param1=hello', 'param2=world'), \
            'build_vset with my_func + param_dict2 + reps fails'
        assert d_keys[1].value == ('rep=1', 'func=my_func', 'param1=hello', 'param2=world'), \
            'build_vset with my_func + param_dict2 + reps fails'
        assert d_keys[2].value == ('rep=0', 'func=my_func', 'param1=hello', 'param2=there'), \
            'build_vset with my_func + param_dict2 + reps fails'
        assert d_keys[3].value == ('rep=1', 'func=my_func', 'param1=hello', 'param2=there'), \
            'build_vset with my_func + param_dict2 + reps fails'
        d_keywords = [val.vfunc.keywords for val in list(vset.vfuncs.values())]
        assert d_keywords[0] == {'param1': 'hello', 'param2': 'world', 'param3': 'b'}, \
            'build_vset with my_func + param_dict2 fails'
        assert d_keywords[1] == {'param1': 'hello', 'param2': 'world', 'param3': 'b'}, \
            'build_vset with my_func + param_dict2 fails'
        assert d_keywords[2] == {'param1': 'hello', 'param2': 'there', 'param3': 'b'}, \
            'build_vset with my_func + param_dict2 fails'
        assert d_keywords[3] == {'param1': 'hello', 'param2': 'there', 'param3': 'b'}, \
            'build_vset with my_func + param_dict2 fails'

        # 1 func with list of param_dicts
        vset = build_vset("vset", my_func, [param_dict1, param_dict2], param3='b')
        assert len(vset) == 5, \
            'build_vset with my_func + [param_dict1, param_dict2] fails'
        d_keys = [key[0].value for key in list(vset.vfuncs.keys())]
        assert ('func=my_func', 'param1=hello', 'param2=world') in d_keys, \
            'build_vset with my_func + [param_dict1, param_dict2] fails'
        assert ('func=my_func', 'param1=hello', 'param2=bar') in d_keys, \
            'build_vset with my_func + [param_dict1, param_dict2] fails'
        assert ('func=my_func', 'param1=foo', 'param2=world') in d_keys, \
            'build_vset with my_func + [param_dict1, param_dict2] fails'
        assert ('func=my_func', 'param1=foo', 'param2=bar') in d_keys, \
            'build_vset with my_func + [param_dict1, param_dict2] fails'
        assert ('func=my_func', 'param1=hello', 'param2=there') in d_keys, \
            'build_vset with my_func + [param_dict1, param_dict2] fails'
        d_keywords = [val.vfunc.keywords for val in list(vset.vfuncs.values())]
        assert {'param1': 'hello', 'param2': 'world', 'param3': 'b'} in d_keywords, \
            'build_vset with my_func + [param_dict1, param_dict2] fails'
        assert {'param1': 'hello', 'param2': 'bar', 'param3': 'b'} in d_keywords, \
            'build_vset with my_func + [param_dict1, param_dict2] fails'
        assert {'param1': 'foo', 'param2': 'world', 'param3': 'b'} in d_keywords, \
            'build_vset with my_func + [param_dict1, param_dict2] fails'
        assert {'param1': 'foo', 'param2': 'bar', 'param3': 'b'} in d_keywords, \
            'build_vset with my_func + [param_dict1, param_dict2] fails'
        assert {'param1': 'hello', 'param2': 'there', 'param3': 'b'} in d_keywords, \
            'build_vset with my_func + [param_dict1, param_dict2] fails'

        # list of funcs with 1 param_dict
        vset = build_vset("vset", [my_func, my_func2], param_dict1, param3='b')
        assert len(vset) == 8, \
            'build_vset with [my_func, my_func2] + param_dict1 fails'
        d_keys = [key[0].value for key in list(vset.vfuncs.keys())]
        assert ('func=my_func', 'param1=hello', 'param2=world') in d_keys, \
            'build_vset with my_func + [param_dict1, param_dict2] fails'
        assert ('func=my_func', 'param1=hello', 'param2=bar') in d_keys, \
            'build_vset with my_func + [param_dict1, param_dict2] fails'
        assert ('func=my_func', 'param1=foo', 'param2=world') in d_keys, \
            'build_vset with my_func + [param_dict1, param_dict2] fails'
        assert ('func=my_func', 'param1=foo', 'param2=bar') in d_keys, \
            'build_vset with my_func + [param_dict1, param_dict2] fails'
        assert ('func=my_func2', 'param1=hello', 'param2=world') in d_keys, \
            'build_vset with my_func + [param_dict1, param_dict2] fails'
        assert ('func=my_func2', 'param1=hello', 'param2=bar') in d_keys, \
            'build_vset with my_func + [param_dict1, param_dict2] fails'
        assert ('func=my_func2', 'param1=foo', 'param2=world') in d_keys, \
            'build_vset with my_func + [param_dict1, param_dict2] fails'
        assert ('func=my_func2', 'param1=foo', 'param2=bar') in d_keys, \
            'build_vset with my_func + [param_dict1, param_dict2] fails'
        d_keywords = [val.vfunc.keywords for val in list(vset.vfuncs.values())]
        assert d_keywords.count({'param1': 'hello', 'param2': 'world', 'param3': 'b'}) == 2, \
            'build_vset with my_func + [param_dict1, param_dict2] fails'
        assert d_keywords.count({'param1': 'hello', 'param2': 'bar', 'param3': 'b'}) == 2, \
            'build_vset with my_func + [param_dict1, param_dict2] fails'
        assert d_keywords.count({'param1': 'foo', 'param2': 'world', 'param3': 'b'}) == 2, \
            'build_vset with my_func + [param_dict1, param_dict2] fails'
        assert d_keywords.count({'param1': 'foo', 'param2': 'bar', 'param3': 'b'}) == 2, \
            'build_vset with my_func + [param_dict1, param_dict2] fails'

        # list of funcs with list of param_dicts 
        vset = build_vset("vset", [my_func, my_func2], [param_dict1, param_dict2])
        assert len(vset) == 6, \
            'build_vset with [my_func, my_func2] + [param_dict1, param_dict2] fails'
        d_keys = [key[0].value for key in list(vset.vfuncs.keys())]
        assert ('func=my_func', 'param1=hello', 'param2=world') in d_keys, \
            'build_vset with my_func + [param_dict1, param_dict2] fails'
        assert ('func=my_func', 'param1=hello', 'param2=bar') in d_keys, \
            'build_vset with my_func + [param_dict1, param_dict2] fails'
        assert ('func=my_func', 'param1=foo', 'param2=world') in d_keys, \
            'build_vset with my_func + [param_dict1, param_dict2] fails'
        assert ('func=my_func', 'param1=foo', 'param2=bar') in d_keys, \
            'build_vset with my_func + [param_dict1, param_dict2] fails'
        assert ('func=my_func2', 'param1=hello', 'param2=world') in d_keys, \
            'build_vset with my_func + [param_dict1, param_dict2] fails'
        assert ('func=my_func2', 'param1=hello', 'param2=there') in d_keys, \
            'build_vset with my_func + [param_dict1, param_dict2] fails'

        class my_class:
            def __init__(self, param1, param2, param3: str='a'):
                self.param1 = param1
                self.param2 = param2
                self.param3 = param3

            def fit(self, arg1: str):
                self.arg1 = arg1

        # my_class without param_dict
        vset = build_vset("vset", my_class, param1='hello', param2='world', param3='b')
        assert len(vset) == 1, \
            'build_vset with my_class fails'
        d_key = [key[0] for key in list(vset.vfuncs.keys())][0]
        assert d_key.value == 'vset_0', \
            'build_vset with my_class fails'
        d_val = [val.vfunc for val in list(vset.vfuncs.values())][0]
        assert isinstance(d_val, my_class), \
            'build_vset with my_class fails'
        assert (d_val.param1, d_val.param2, d_val.param3) == ('hello', 'world', 'b'), \
            'build_vset with my_class fails'

        # my_class without param_dict, reps
        vset = build_vset("vset", my_class, reps=2, param1='hello', param2='world', param3='b')
        assert len(vset) == 2, \
            'build_vset with my_class + reps fails'
        d_keys = [key[0].value[0] for key in list(vset.vfuncs.keys())]
        assert d_keys[0] == 'rep=0', \
            'build_vset with my_class + reps fails'
        assert d_keys[1] == 'rep=1', \
            'build_vset with my_class + reps fails'
        d_vals = [val.vfunc for val in list(vset.vfuncs.values())]
        assert isinstance(d_vals[0], my_class), \
            'build_vset with my_class + reps fails'
        assert isinstance(d_vals[1], my_class), \
            'build_vset with my_class + reps fails'
        assert (d_vals[0].param1, d_vals[0].param2, d_vals[0].param3) == ('hello', 'world', 'b'), \
            'build_vset with my_class + reps fails'
        assert (d_vals[1].param1, d_vals[1].param2, d_vals[1].param3) == ('hello', 'world', 'b'), \
            'build_vset with my_class + reps fails'

        # my_class with param_dict1
        vset = build_vset("vset", my_class, param_dict1, param3='b')
        assert len(vset) == 4, \
            'build_vset with my_class + param_dict1 fails'
        d_keys = [key[0] for key in list(vset.vfuncs.keys())]
        assert d_keys[0].value == ('func=my_class', 'param1=hello', 'param2=world'), \
            'build_vset with my_class + param_dict1 fails'
        assert d_keys[1].value == ('func=my_class', 'param1=hello', 'param2=bar'), \
            'build_vset with my_class + param_dict1 fails'
        assert d_keys[2].value == ('func=my_class', 'param1=foo', 'param2=world'), \
            'build_vset with my_class + param_dict1 fails'
        assert d_keys[3].value == ('func=my_class', 'param1=foo', 'param2=bar'), \
            'build_vset with my_class + param_dict1 fails'
        d_vals = [val.vfunc for val in list(vset.vfuncs.values())]
        assert isinstance(d_vals[0], my_class), \
            'build_vset with my_class + param_dict1 fails'
        assert isinstance(d_vals[1], my_class), \
            'build_vset with my_class + param_dict1 fails'
        assert isinstance(d_vals[1], my_class), \
            'build_vset with my_class + param_dict1 fails'
        assert isinstance(d_vals[1], my_class), \
            'build_vset with my_class + param_dict1 fails'
        assert (d_vals[0].param1, d_vals[0].param2, d_vals[0].param3) == ('hello', 'world', 'b'), \
            'build_vset with my_class + param_dict1 fails'
        assert (d_vals[1].param1, d_vals[1].param2, d_vals[1].param3) == ('hello', 'bar', 'b'), \
            'build_vset with my_class + param_dict1 fails'
        assert (d_vals[2].param1, d_vals[2].param2, d_vals[2].param3) == ('foo', 'world', 'b'), \
            'build_vset with my_class + param_dict1 fails'
        assert (d_vals[3].param1, d_vals[3].param2, d_vals[3].param3) == ('foo', 'bar', 'b'), \
            'build_vset with my_class + param_dict1 fails'

        # my_class with param_dict2, reps
        vset = build_vset("vset", my_class, param_dict2, reps=2, lazy=True, param3='b')
        assert vset._lazy, \
            'build_vset with my_class + param_dict2 + reps fails'
        assert len(vset) == 4, \
            'build_vset with my_class + param_dict2 + reps fails'
        d_keys = [key[0] for key in list(vset.vfuncs.keys())]
        assert d_keys[0].value == ('rep=0', 'func=my_class', 'param1=hello', 'param2=world'), \
            'build_vset with my_class + param_dict2 + reps fails'
        assert d_keys[1].value == ('rep=1', 'func=my_class', 'param1=hello', 'param2=world'), \
            'build_vset with my_class + param_dict2 + reps fails'
        assert d_keys[2].value == ('rep=0', 'func=my_class', 'param1=hello', 'param2=there'), \
            'build_vset with my_class + param_dict2 + reps fails'
        assert d_keys[3].value == ('rep=1', 'func=my_class', 'param1=hello', 'param2=there'), \
            'build_vset with my_class + param_dict2 + reps fails'
        d_vals = [val.vfunc for val in list(vset.vfuncs.values())]
        assert isinstance(d_vals[0], my_class), \
            'build_vset with my_class + param_dict2 + reps fails'
        assert isinstance(d_vals[1], my_class), \
            'build_vset with my_class + param_dict2 + reps fails'
        assert isinstance(d_vals[1], my_class), \
            'build_vset with my_class + param_dict2 + reps fails'
        assert isinstance(d_vals[1], my_class), \
            'build_vset with my_class + param_dict2 + reps fails'
        assert (d_vals[0].param1, d_vals[0].param2, d_vals[0].param3) == ('hello', 'world', 'b'), \
            'build_vset with my_class + param_dict2 + reps fails'
        assert (d_vals[1].param1, d_vals[1].param2, d_vals[1].param3) == ('hello', 'world', 'b'), \
            'build_vset with my_class + param_dict2 + reps fails'
        assert (d_vals[2].param1, d_vals[2].param2, d_vals[2].param3) == ('hello', 'there', 'b'), \
            'build_vset with my_class + param_dict2 + reps fails'
        assert (d_vals[3].param1, d_vals[3].param2, d_vals[3].param3) == ('hello', 'there', 'b'), \
            'build_vset with my_class + param_dict2 + reps fails'


    def test_cum_acc_by_uncertainty(self):
        mean_dict = {'group_0': np.array([[0.2, 0.8], [0.25, 0.75], [0.1, 0.9]]),
                     'group_1': np.array([[0.4, 0.6], [0.5, 0.5], [0.45, 0.55]])}
        std_dict = {'group_0': np.array([[0.003, 0.003], [0.146, 0.146], [0.0023, 0.0023]]),
                    'group_1': np.array([[0.0054, 0.0054], [0.2344, 0.2344], [0.5166, 0.5166]])}
        true_labels = [0, 1, 1]
        true_labels_dict = {'y': [0, 1, 1]}
        u0, c0, idx0 = cum_acc_by_uncertainty(mean_dict, std_dict, true_labels)
        u1, c1, idx1 = cum_acc_by_uncertainty(mean_dict, std_dict, true_labels_dict)
        assert_equal(u0, u1)
        assert_equal(c0, c1)
        assert_equal(idx0, idx1)
        assert u0.shape == c0.shape == (2, 3)
        assert_equal(u0[0], sorted(x[1] for x in std_dict['group_0']))
        assert_equal(u0[1], sorted(x[1] for x in std_dict['group_1']))
        assert_equal(c0[0], [1, 1/2, 2/3])
        assert_equal(c0[1], [0, 0, 1/3])
        assert_equal(idx0[0], [2, 0, 1])
        assert_equal(idx0[1], [0, 1, 2])
