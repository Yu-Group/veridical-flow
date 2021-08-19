import pytest

import pcsp
from pcsp.module_set import PREV_KEY, MATCH_KEY
from pcsp.convert import *

import numpy as np
from numpy.testing import assert_equal

@pytest.mark.parametrize(
    'in_dicts,out_dict',
    [
        # first or second dict has only one key
        (
            # in_dicts
            [
                {
                    ('X_train','y_train','RF'):'RF_fitted',
                    ('X_train','y_train','LR'):'LR_fitted',
                    PREV_KEY:('prev_0',)
                },
                {('X_test',):'X_test_data', PREV_KEY:('prev_1',)}
            ],
            # out_dict
            {
                PREV_KEY:('prev_0','prev_1',), MATCH_KEY:0,
                ('X_train','y_train','RF','X_test'):('RF_fitted','X_test_data'),
                ('X_train','y_train','LR','X_test'):('LR_fitted','X_test_data')
            }
        ),
        (
            # in_dicts
            [
                {
                    PREV_KEY:('prev_0','prev_1',), MATCH_KEY:0,
                    ('X_train','y_train','RF','X_test'):['RF_fitted','X_test_data'],
                    ('X_train','y_train','LR','X_test'):['LR_fitted','X_test_data']
                },
                {
                    ('y_test',):'y_test_data', ('y_test',):'y_test_data',
                    PREV_KEY:('prev_2',),MATCH_KEY:0
                }
            ],
            # out_dict
            {
                PREV_KEY:('prev_0','prev_1','prev_2',),
                MATCH_KEY:0,
                ('X_train','y_train','RF','X_test','y_test'):(
                    ['RF_fitted','X_test_data'],'y_test_data'
                ),
                ('X_train','y_train','LR','X_test','y_test'):(
                    ['LR_fitted','X_test_data'],'y_test_data'
                )
            }
        ),
        (
            # in_dicts
            [
                {('X_train','y_train','subsampling_0','RF'):'RF_fitted_0',
                 ('X_train','y_train','subsampling_1','RF'):'RF_fitted_1',
                 ('X_train','y_train','subsampling_0','LR'):'LR_fitted_0',
                 ('X_train','y_train','subsampling_1','LR'):'LR_fitted_1'},
                {('X_test',):'X_test_data'},
                {('y_test',):'y_test_data'}

            ],
            # out_dict
            {
                ('X_train','y_train','subsampling_0','RF','X_test','y_test'):(
                    'RF_fitted_0','X_test_data','y_test_data'
                ),
                ('X_train','y_train','subsampling_1','RF','X_test','y_test'):(
                    'RF_fitted_1','X_test_data','y_test_data'
                ),
                ('X_train','y_train','subsampling_0','LR','X_test','y_test'):(
                    'LR_fitted_0','X_test_data','y_test_data'
                ),
                ('X_train','y_train','subsampling_1','LR','X_test','y_test'):(
                    'LR_fitted_1','X_test_data','y_test_data'
                ),
                PREV_KEY:(),
                MATCH_KEY:0
            }
        ),
        (
            # in_dicts
            [
                {
                    ('X_train','y_train','subgroup_0','voxel_extract_0','RF'):'RF_fitted_00',
                    ('X_train','y_train','subgroup_0','voxel_extract_1','RF'):'RF_fitted_01',
                    ('X_train','y_train','subgroup_1','voxel_extract_0','RF'):'RF_fitted_10',
                    ('X_train','y_train','subgroup_1','voxel_extract_1','RF'):'RF_fitted_11',
                    ('X_train','y_train','subgroup_0','voxel_extract_0','LR'):'LR_fitted_00',
                    ('X_train','y_train','subgroup_0','voxel_extract_1','LR'):'LR_fitted_01',
                    ('X_train','y_train','subgroup_1','voxel_extract_0','LR'):'LR_fitted_10',
                    ('X_train','y_train','subgroup_1','voxel_extract_1','LR'):'LR_fitted_11'
                },
                {
                    MATCH_KEY:2,
                    ('X_test','subgroup_0','voxel_extract_0'):'X_test_data_00',
                    ('X_test','subgroup_0','voxel_extract_1'):'X_test_data_01',
                    ('X_test','subgroup_1','voxel_extract_0'):'X_test_data_10',
                    ('X_test','subgroup_1','voxel_extract_1'):'X_test_data_11'
                }
            ],
            # out_dict
            {
                ('X_train','y_train','subgroup_0','voxel_extract_0','RF','X_test'):(
                    'RF_fitted_00','X_test_data_00'
                ),
                ('X_train','y_train','subgroup_0','voxel_extract_1','RF','X_test'):(
                    'RF_fitted_01','X_test_data_01'
                ),
                ('X_train','y_train','subgroup_1','voxel_extract_0','RF','X_test'):(
                    'RF_fitted_10','X_test_data_10'
                ),
                ('X_train','y_train','subgroup_1','voxel_extract_1','RF','X_test'):(
                    'RF_fitted_11','X_test_data_11'
                ),
                ('X_train','y_train','subgroup_0','voxel_extract_0','LR','X_test'):(
                    'LR_fitted_00','X_test_data_00'
                ),
                ('X_train','y_train','subgroup_0','voxel_extract_1','LR','X_test'):(
                    'LR_fitted_01','X_test_data_01'
                ),
                ('X_train','y_train','subgroup_1','voxel_extract_0','LR','X_test'):(
                    'LR_fitted_10','X_test_data_10'
                ),
                ('X_train','y_train','subgroup_1','voxel_extract_1','LR','X_test'):(
                    'LR_fitted_11','X_test_data_11'
                ),
                PREV_KEY:(),
                MATCH_KEY:0
            }
        ),
        (
            # in_dicts
            [
                {
                    ('X_train','y_train','subgroup_0','voxel_extract_0','RF','X_test'):[
                        'RF_fitted_00','X_test_data_00'
                    ],
                    ('X_train','y_train','subgroup_0','voxel_extract_1','RF','X_test'):[
                        'RF_fitted_01','X_test_data_01'
                    ],
                    ('X_train','y_train','subgroup_1','voxel_extract_0','RF','X_test'):[
                        'RF_fitted_10','X_test_data_10'
                    ],
                    ('X_train','y_train','subgroup_1','voxel_extract_1','RF','X_test'):[
                        'RF_fitted_11','X_test_data_11'
                    ],
                    ('X_train','y_train','subgroup_0','voxel_extract_0','LR','X_test'):[
                        'LR_fitted_00','X_test_data_00'
                    ],
                    ('X_train','y_train','subgroup_0','voxel_extract_1','LR','X_test'):[
                        'LR_fitted_01','X_test_data_01'
                    ],
                    ('X_train','y_train','subgroup_1','voxel_extract_0','LR','X_test'):[
                        'LR_fitted_10','X_test_data_10'
                    ],
                    ('X_train','y_train','subgroup_1','voxel_extract_1','LR','X_test'):[
                        'LR_fitted_11','X_test_data_11'
                    ],
                    PREV_KEY:(),
                    MATCH_KEY:0
                },
                {
                    MATCH_KEY:2,
                    ('y_test','subgroup_0','voxel_extract_0'):'y_test_data_00',
                    ('y_test','subgroup_0','voxel_extract_1'):'y_test_data_01',
                    ('y_test','subgroup_1','voxel_extract_0'):'y_test_data_10',
                    ('y_test','subgroup_1','voxel_extract_1'):'y_test_data_11'
                }
            ],
            # out_dict
            {
                ('X_train','y_train','subgroup_0','voxel_extract_0','RF','X_test','y_test'):(
                    ['RF_fitted_00','X_test_data_00'],'y_test_data_00'
                ),
                ('X_train','y_train','subgroup_0','voxel_extract_1','RF','X_test','y_test'):(
                    ['RF_fitted_01','X_test_data_01'],'y_test_data_01'
                ),
                ('X_train','y_train','subgroup_1','voxel_extract_0','RF','X_test','y_test'):(
                    ['RF_fitted_10','X_test_data_10'],'y_test_data_10'
                ),
                ('X_train','y_train','subgroup_1','voxel_extract_1','RF','X_test','y_test'):(
                    ['RF_fitted_11','X_test_data_11'],'y_test_data_11'
                ),
                ('X_train','y_train','subgroup_0','voxel_extract_0','LR','X_test','y_test'):(
                    ['LR_fitted_00','X_test_data_00'],'y_test_data_00'
                ),
                ('X_train','y_train','subgroup_0','voxel_extract_1','LR','X_test','y_test'):(
                    ['LR_fitted_01','X_test_data_01'],'y_test_data_01'
                ),
                ('X_train','y_train','subgroup_1','voxel_extract_0','LR','X_test','y_test'):(
                    ['LR_fitted_10','X_test_data_10'],'y_test_data_10'
                ),
                ('X_train','y_train','subgroup_1','voxel_extract_1','LR','X_test','y_test'):(
                    ['LR_fitted_11','X_test_data_11'],'y_test_data_11'
                ),
                PREV_KEY:(),
                MATCH_KEY:0
            }
        ),
        (
            # in_dicts
            [
                {
                    ('X_train','y_train','subgroup_0','voxel_extract_0','RF','X_test','y_test'):[
                        ['RF_fitted_00','X_test_data_00'],'y_test_data'
                    ],
                    ('X_train','y_train','subgroup_0','voxel_extract_1','RF','X_test','y_test'):[
                        ['RF_fitted_01','X_test_data_01'],'y_test_data'
                    ],
                    ('X_train','y_train','subgroup_1','voxel_extract_0','RF','X_test','y_test'):[
                        ['RF_fitted_10','X_test_data_10'],'y_test_data'
                    ],
                    ('X_train','y_train','subgroup_1','voxel_extract_1','RF','X_test','y_test'):[
                        ['RF_fitted_11','X_test_data_11'],'y_test_data'
                    ],
                    ('X_train','y_train','subgroup_0','voxel_extract_0','LR','X_test','y_test'):[
                        ['LR_fitted_00','X_test_data_00'],'y_test_data'
                    ],
                    ('X_train','y_train','subgroup_0','voxel_extract_1','LR','X_test','y_test'):[
                        ['LR_fitted_01','X_test_data_01'],'y_test_data'
                    ],
                    ('X_train','y_train','subgroup_1','voxel_extract_0','LR','X_test','y_test'):[
                        ['LR_fitted_10','X_test_data_10'],'y_test_data'
                    ],
                    ('X_train','y_train','subgroup_1','voxel_extract_1','LR','X_test','y_test'):[
                        ['LR_fitted_11','X_test_data_11'],'y_test_data'
                    ],
                    PREV_KEY:(),
                    MATCH_KEY:0
                },
                {
                    MATCH_KEY:1,
                    ('LR','acc'):'LR_acc_func',
                    ('LR','bal_acc'):'LR_bal_acc_func',
                    ('RF','acc'):'RF_acc_func',
                    ('RF','feat_imp'):'RF_feat_imp_func'
                }
            ],
            # out_dict
            {
                ('X_train','y_train','subgroup_0','voxel_extract_0','RF','X_test','y_test','acc'):(
                    [['RF_fitted_00','X_test_data_00'],'y_test_data'],'RF_acc_func'
                ),
                ('X_train','y_train','subgroup_0','voxel_extract_0','RF','X_test','y_test','feat_imp'):(
                    [['RF_fitted_00','X_test_data_00'],'y_test_data'],'RF_feat_imp_func'
                ),
                ('X_train','y_train','subgroup_0','voxel_extract_1','RF','X_test','y_test','acc'):(
                    [['RF_fitted_01','X_test_data_01'],'y_test_data'],'RF_acc_func'
                ),
                ('X_train','y_train','subgroup_0','voxel_extract_1','RF','X_test','y_test','feat_imp'):(
                    [['RF_fitted_01','X_test_data_01'],'y_test_data'],'RF_feat_imp_func'
                ),
                ('X_train','y_train','subgroup_1','voxel_extract_0','RF','X_test','y_test','acc'):(
                    [['RF_fitted_10','X_test_data_10'],'y_test_data'],'RF_acc_func'
                ),
                ('X_train','y_train','subgroup_1','voxel_extract_0','RF','X_test','y_test','feat_imp'):(
                    [['RF_fitted_10','X_test_data_10'],'y_test_data'],'RF_feat_imp_func'
                ),
                ('X_train','y_train','subgroup_1','voxel_extract_1','RF','X_test','y_test','acc'):(
                    [['RF_fitted_11','X_test_data_11'],'y_test_data'],'RF_acc_func'
                ),
                ('X_train','y_train','subgroup_1','voxel_extract_1','RF','X_test','y_test','feat_imp'):(
                    [['RF_fitted_11','X_test_data_11'],'y_test_data'],'RF_feat_imp_func'
                ),
                ('X_train','y_train','subgroup_0','voxel_extract_0','LR','X_test','y_test','acc'):(
                    [['LR_fitted_00','X_test_data_00'],'y_test_data'],'LR_acc_func'
                ),
                ('X_train','y_train','subgroup_0','voxel_extract_0','LR','X_test','y_test','bal_acc'):(
                    [['LR_fitted_00','X_test_data_00'],'y_test_data'],'LR_bal_acc_func'
                ),
                ('X_train','y_train','subgroup_0','voxel_extract_1','LR','X_test','y_test','acc'):(
                    [['LR_fitted_01','X_test_data_01'],'y_test_data'],'LR_acc_func'
                ),
                ('X_train','y_train','subgroup_0','voxel_extract_1','LR','X_test','y_test','bal_acc'):(
                    [['LR_fitted_01','X_test_data_01'],'y_test_data'],'LR_bal_acc_func'
                ),
                ('X_train','y_train','subgroup_1','voxel_extract_0','LR','X_test','y_test','acc'):(
                    [['LR_fitted_10','X_test_data_10'],'y_test_data'],'LR_acc_func'
                ),
                ('X_train','y_train','subgroup_1','voxel_extract_0','LR','X_test','y_test','bal_acc'):(
                    [['LR_fitted_10','X_test_data_10'],'y_test_data'],'LR_bal_acc_func'
                ),
                ('X_train','y_train','subgroup_1','voxel_extract_1','LR','X_test','y_test','acc'):(
                    [['LR_fitted_11','X_test_data_11'],'y_test_data'],'LR_acc_func'
                ),
                ('X_train','y_train','subgroup_1','voxel_extract_1','LR','X_test','y_test','bal_acc'):(
                    [['LR_fitted_11','X_test_data_11'],'y_test_data'],'LR_bal_acc_func'
                ),
                PREV_KEY:(),
                MATCH_KEY:0
            }
        ),
        pytest.param(
            # in_dicts
            [
                {
                    PREV_KEY: 'prev_0' # not wrapped in tuple
                },
                {
                    PREV_KEY: ('prev_1', )
                }
            ],
            # out_dict
            {
                PREV_KEY: ('prev_0', 'prev_1')
            },
            # this test is expected to fail because combine_dicts() makes the
            # assumption that PREV_KEY entries are wrapped in tuples
            marks=pytest.mark.xfail
        ),
    ]
)
class TestCombineDicts:

    def test_combine_dicts(self, in_dicts, out_dict):
        result_dict = combine_dicts(*in_dicts)
        assert_equal(out_dict, result_dict)


@pytest.mark.parametrize(
    'in_dicts,out_dict',
    [
        (
            # in_dicts
            [
                # modules
                {
                    ('module_0',): lambda x, y: x+y,
                    ('module_1',): lambda x, y: x*y,
                },
                # data_dict
                {('data',): [2, 3]}
            ],
            # out_dict
            {
                ('data', 'module_0'): 5,
                ('data', 'module_1'): 6
            }
        ),
        (
            # in_dicts
            [
                # modules
                {
                    ('group_0','module_0',): lambda x, y: x+y,
                    ('group_1','module_1',): lambda x, y: x*y,
                },
                # data_dict
                {
                    MATCH_KEY: 1,
                    ('data','group_0'): [np.array([1,2,3]), np.array([4,5,6])],
                    ('data','group_1'): [np.array([1,2,3]), np.array([4,5,6])],
                }
            ],
            # out_dict
            {
                ('data','group_0','module_0'): np.array([5,7,9]),
                ('data','group_1','module_1'): np.array([4,10,18]),
            }
        ),
        pytest.param(
            # in_dicts
            [
                # modules
                {
                    ('data','group_0','module_0',): lambda x, y: x+y,
                    ('data','group_1','module_1',): lambda x, y: x*y,
                },
                # data_dict
                {
                    MATCH_KEY: 1,
                    ('data','group_0'): [np.array([1,2,3]), np.array([4,5,6])],
                    ('data','group_1'): [np.array([1,2,3]), np.array([4,5,6])],
                }
            ],
            # out_dict
            {
                ('data','group_0','module_0'): np.array([5,7,9]),
                ('data','group_1','module_1'): np.array([4,10,18]),
            },
            marks=pytest.mark.xfail
        ),
    ]
)
class TestApplyModules:

    def test_apply_modules(self, in_dicts, out_dict):
        result_dict = apply_modules(*in_dicts)
        assert_equal(out_dict, result_dict)
