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
                PREV_KEY:('prev_0','prev_1',), MATCH_KEY:[],
                ('X_train','y_train','RF','X_test'):('RF_fitted','X_test_data'),
                ('X_train','y_train','LR','X_test'):('LR_fitted','X_test_data')
            }
        ),
        (
            # in_dicts
            [
                {
                    PREV_KEY:('prev_0','prev_1',), MATCH_KEY:[],
                    ('X_train','y_train','RF','X_test'):['RF_fitted','X_test_data'],
                    ('X_train','y_train','LR','X_test'):['LR_fitted','X_test_data']
                },
                {
                    ('y_test',):'y_test_data', ('y_test',):'y_test_data',
                    PREV_KEY:('prev_2',),MATCH_KEY:[]
                }
            ],
            # out_dict
            {
                PREV_KEY:('prev_0','prev_1','prev_2',),
                MATCH_KEY:[],
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
                MATCH_KEY:[]
            }
        ),
        (
            # in_dicts
            [
                {
                    MATCH_KEY:[2],
                    ('X_train','y_train','subsampling_0','RF'):'RF_fitted_0',
                    ('X_train','y_train','subsampling_1','RF'):'RF_fitted_1',
                    ('X_train','y_train','subsampling_0','LR'):'LR_fitted_0',
                    ('X_train','y_train','subsampling_1','LR'):'LR_fitted_1',
                },
                {
                    ('X_train','subsampling_0'):'X_train_data_0',
                    ('X_train','subsampling_1'):'X_train_data_1',
                    MATCH_KEY:[1]
                },
                {
                    ('y_train','subsampling_0'):'y_train_data_0',
                    ('y_train','subsampling_1'):'y_train_data_1',
                    MATCH_KEY:[1]
                }
            ],
            # out_dict
            {
                ('X_train','y_train','subsampling_0','RF','X_train','y_train'):(
                    'RF_fitted_0','X_train_data_0','y_train_data_0'
                ),
                ('X_train','y_train','subsampling_1','RF','X_train','y_train'):(
                    'RF_fitted_1','X_train_data_1','y_train_data_1'
                ),
                ('X_train','y_train','subsampling_0','LR','X_train','y_train'):(
                    'LR_fitted_0','X_train_data_0','y_train_data_0'
                ),
                ('X_train','y_train','subsampling_1','LR','X_train','y_train'):(
                    'LR_fitted_1','X_train_data_1','y_train_data_1'
                ),
                PREV_KEY:(),
                MATCH_KEY:[2]
            }
        ),
        (
            # in_dicts
            [
                {
                    MATCH_KEY:[2,3],
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
                    MATCH_KEY:[1,2],
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
                MATCH_KEY:[2,3]
            }
        ),
        (
            # in_dicts
            [
                {
                    MATCH_KEY:[2,3],
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
                    MATCH_KEY:[1,2],
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
                MATCH_KEY:[2,3]
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
                    MATCH_KEY:[2,3]
                },
                {
                    MATCH_KEY:[1,2],
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
                MATCH_KEY:[2,3]
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
                    MATCH_KEY:[2,3]
                },
                {
                    MATCH_KEY:[0],
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
                MATCH_KEY:[2,3,4]
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
            marks=pytest.mark.xfail(strict=True)
        ),
        (
            # in_dicts
            [
                {
                    MATCH_KEY:[2,3],
                    ('X_train','y_train','voxel_extract_0','subgroup_0', 'RF'):'RF_00',
                    ('X_train','y_train','voxel_extract_0','subgroup_1', 'RF'):'RF_01',
                    ('X_train','y_train','voxel_extract_1','subgroup_0', 'RF'):'RF_10',
                    ('X_train','y_train','voxel_extract_1','subgroup_1', 'RF'):'RF_11',
                },
                {
                    MATCH_KEY:[1],
                    ('X_test','subgroup_0'):'X_test_data_0',
                    ('X_test','subgroup_1'):'X_test_data_1'
                },
                {
                    MATCH_KEY:[1,2],
                    ('y_test','voxel_extract_0','subgroup_0'):'y_test_data_00',
                    ('y_test','voxel_extract_0','subgroup_1'):'y_test_data_01',
                    ('y_test','voxel_extract_1','subgroup_0'):'y_test_data_10',
                    ('y_test','voxel_extract_1','subgroup_1'):'y_test_data_11',
                }
            ],
            # out_dict
            {
                MATCH_KEY:[2,3],
                PREV_KEY:(),
                ('X_train','y_train','voxel_extract_0','subgroup_0','RF','X_test','y_test'):(
                    'RF_00', 'X_test_data_0', 'y_test_data_00'
                ),
                ('X_train','y_train','voxel_extract_0','subgroup_1','RF','X_test','y_test'):(
                    'RF_01', 'X_test_data_1', 'y_test_data_01'
                ),
                ('X_train','y_train','voxel_extract_1','subgroup_0','RF','X_test','y_test'):(
                    'RF_10', 'X_test_data_0', 'y_test_data_10'
                ),
                ('X_train','y_train','voxel_extract_1','subgroup_1','RF','X_test','y_test'):(
                    'RF_11', 'X_test_data_1', 'y_test_data_11'
                )
            },
        ),
        (
            # in_dicts
            [
                {
                    MATCH_KEY:[2,3],
                    ('X_train','y_train','voxel_extract_0','subgroup_0', 'RF'):'RF_00',
                    ('X_train','y_train','voxel_extract_0','subgroup_1', 'RF'):'RF_01',
                    ('X_train','y_train','voxel_extract_1','subgroup_0', 'RF'):'RF_10',
                    ('X_train','y_train','voxel_extract_1','subgroup_1', 'RF'):'RF_11',
                },
                {
                    MATCH_KEY:[1,2],
                    ('y_test','voxel_extract_0','subgroup_0'):'y_test_data_00',
                    ('y_test','voxel_extract_0','subgroup_1'):'y_test_data_01',
                    ('y_test','voxel_extract_1','subgroup_0'):'y_test_data_10',
                    ('y_test','voxel_extract_1','subgroup_1'):'y_test_data_11',
                },
                {
                    MATCH_KEY:[1],
                    ('X_test','subgroup_0'):'X_test_data_0',
                    ('X_test','subgroup_1'):'X_test_data_1'
                },
            ],
            # out_dict
            {
                MATCH_KEY:[2,3],
                PREV_KEY:(),
                ('X_train','y_train','voxel_extract_0','subgroup_0','RF','y_test','X_test'):(
                    'RF_00', 'y_test_data_00', 'X_test_data_0'
                ),
                ('X_train','y_train','voxel_extract_0','subgroup_1','RF','y_test','X_test'):(
                    'RF_01', 'y_test_data_01', 'X_test_data_1'
                ),
                ('X_train','y_train','voxel_extract_1','subgroup_0','RF','y_test','X_test'):(
                    'RF_10', 'y_test_data_10', 'X_test_data_0'
                ),
                ('X_train','y_train','voxel_extract_1','subgroup_1','RF','y_test','X_test'):(
                    'RF_11', 'y_test_data_11', 'X_test_data_1'
                )
            },
        ),
        (
            # in_dicts
            [
                {
                    MATCH_KEY:[1],
                    ('X_test','subgroup_0'):'X_test_data_0',
                    ('X_test','subgroup_1'):'X_test_data_1'
                },
                {
                    MATCH_KEY:[2,3],
                    ('X_train','y_train','voxel_extract_0','subgroup_0', 'RF'):'RF_00',
                    ('X_train','y_train','voxel_extract_0','subgroup_1', 'RF'):'RF_01',
                    ('X_train','y_train','voxel_extract_1','subgroup_0', 'RF'):'RF_10',
                    ('X_train','y_train','voxel_extract_1','subgroup_1', 'RF'):'RF_11',
                },
                {
                    MATCH_KEY:[1,2],
                    ('y_test','voxel_extract_0','subgroup_0'):'y_test_data_00',
                    ('y_test','voxel_extract_0','subgroup_1'):'y_test_data_01',
                    ('y_test','voxel_extract_1','subgroup_0'):'y_test_data_10',
                    ('y_test','voxel_extract_1','subgroup_1'):'y_test_data_11',
                }
            ],
            # out_dict
            {
                MATCH_KEY:[1,4],
                PREV_KEY:(),
                ('X_test','subgroup_0','X_train','y_train','voxel_extract_0','RF','y_test'):(
                    'X_test_data_0', 'RF_00', 'y_test_data_00'
                ),
                ('X_test','subgroup_1','X_train','y_train','voxel_extract_0','RF','y_test'):(
                    'X_test_data_1', 'RF_01', 'y_test_data_01'
                ),
                ('X_test','subgroup_0','X_train','y_train','voxel_extract_1','RF','y_test'):(
                    'X_test_data_0', 'RF_10', 'y_test_data_10'
                ),
                ('X_test','subgroup_1','X_train','y_train','voxel_extract_1','RF','y_test'):(
                    'X_test_data_1', 'RF_11', 'y_test_data_11'
                )
            },
        ),
        (
            # in_dicts
            [
                {
                    MATCH_KEY:[1],
                    ('X_test','subgroup_0'):'X_test_data_0',
                    ('X_test','subgroup_1'):'X_test_data_1'
                },
                {
                    MATCH_KEY:[1,2],
                    ('y_test','voxel_extract_0','subgroup_0'):'y_test_data_00',
                    ('y_test','voxel_extract_0','subgroup_1'):'y_test_data_01',
                    ('y_test','voxel_extract_1','subgroup_0'):'y_test_data_10',
                    ('y_test','voxel_extract_1','subgroup_1'):'y_test_data_11',
                },
                {
                    MATCH_KEY:[2,3],
                    ('X_train','y_train','voxel_extract_0','subgroup_0', 'RF'):'RF_00',
                    ('X_train','y_train','voxel_extract_0','subgroup_1', 'RF'):'RF_01',
                    ('X_train','y_train','voxel_extract_1','subgroup_0', 'RF'):'RF_10',
                    ('X_train','y_train','voxel_extract_1','subgroup_1', 'RF'):'RF_11',
                },
            ],
            # out_dict
            {
                MATCH_KEY:[1,3],
                PREV_KEY:(),
                ('X_test','subgroup_0','y_test','voxel_extract_0','X_train','y_train','RF'):(
                    'X_test_data_0', 'y_test_data_00', 'RF_00'
                ),
                ('X_test','subgroup_1','y_test','voxel_extract_0','X_train','y_train','RF'):(
                    'X_test_data_1', 'y_test_data_01', 'RF_01'
                ),
                ('X_test','subgroup_0','y_test','voxel_extract_1','X_train','y_train','RF'):(
                    'X_test_data_0', 'y_test_data_10', 'RF_10'
                ),
                ('X_test','subgroup_1','y_test','voxel_extract_1','X_train','y_train','RF'):(
                    'X_test_data_1', 'y_test_data_11', 'RF_11'
                )
            },
        ),
        (
            # in_dicts
            [
                {
                    MATCH_KEY:[1,2],
                    ('y_test','voxel_extract_0','subgroup_0'):'y_test_data_00',
                    ('y_test','voxel_extract_0','subgroup_1'):'y_test_data_01',
                    ('y_test','voxel_extract_1','subgroup_0'):'y_test_data_10',
                    ('y_test','voxel_extract_1','subgroup_1'):'y_test_data_11',
                },
                {
                    MATCH_KEY:[2,3],
                    ('X_train','y_train','voxel_extract_0','subgroup_0', 'RF'):'RF_00',
                    ('X_train','y_train','voxel_extract_0','subgroup_1', 'RF'):'RF_01',
                    ('X_train','y_train','voxel_extract_1','subgroup_0', 'RF'):'RF_10',
                    ('X_train','y_train','voxel_extract_1','subgroup_1', 'RF'):'RF_11',
                },
                {
                    MATCH_KEY:[1],
                    ('X_test','subgroup_0'):'X_test_data_0',
                    ('X_test','subgroup_1'):'X_test_data_1'
                },
            ],
            # out_dict
            {
                MATCH_KEY:[1,2],
                PREV_KEY:(),
                ('y_test','voxel_extract_0','subgroup_0','X_train','y_train','RF','X_test'):(
                    'y_test_data_00', 'RF_00', 'X_test_data_0'
                ),
                ('y_test','voxel_extract_0','subgroup_1','X_train','y_train','RF','X_test'):(
                    'y_test_data_01', 'RF_01', 'X_test_data_1'
                ),
                ('y_test','voxel_extract_1','subgroup_0','X_train','y_train','RF','X_test'):(
                    'y_test_data_10', 'RF_10', 'X_test_data_0'
                ),
                ('y_test','voxel_extract_1','subgroup_1','X_train','y_train','RF','X_test'):(
                    'y_test_data_11', 'RF_11', 'X_test_data_1'
                )
            },
        ),
        (
            # in_dicts
            [
                {
                    MATCH_KEY:[1,2],
                    ('y_test','voxel_extract_0','subgroup_0'):'y_test_data_00',
                    ('y_test','voxel_extract_0','subgroup_1'):'y_test_data_01',
                    ('y_test','voxel_extract_1','subgroup_0'):'y_test_data_10',
                    ('y_test','voxel_extract_1','subgroup_1'):'y_test_data_11',
                },
                {
                    MATCH_KEY:[1],
                    ('X_test','subgroup_0'):'X_test_data_0',
                    ('X_test','subgroup_1'):'X_test_data_1'
                },
                {
                    MATCH_KEY:[2,3],
                    ('X_train','y_train','voxel_extract_0','subgroup_0', 'RF'):'RF_00',
                    ('X_train','y_train','voxel_extract_0','subgroup_1', 'RF'):'RF_01',
                    ('X_train','y_train','voxel_extract_1','subgroup_0', 'RF'):'RF_10',
                    ('X_train','y_train','voxel_extract_1','subgroup_1', 'RF'):'RF_11',
                },
            ],
            # out_dict
            {
                MATCH_KEY:[1,2],
                PREV_KEY:(),
                ('y_test','voxel_extract_0','subgroup_0','X_test','X_train','y_train','RF'):(
                    'y_test_data_00', 'X_test_data_0', 'RF_00'
                ),
                ('y_test','voxel_extract_0','subgroup_1','X_test','X_train','y_train','RF'):(
                    'y_test_data_01', 'X_test_data_1', 'RF_01'
                ),
                ('y_test','voxel_extract_1','subgroup_0','X_test','X_train','y_train','RF'):(
                    'y_test_data_10', 'X_test_data_0', 'RF_10'
                ),
                ('y_test','voxel_extract_1','subgroup_1','X_test','X_train','y_train','RF'):(
                    'y_test_data_11', 'X_test_data_1', 'RF_11'
                )
            },
        ),
        pytest.param(
            # in_dicts
            [
                {
                    MATCH_KEY:[1,2],
                    ('X_test','feature_extraction_0', 'subgroup_0'):'X_test_data_00',
                    ('X_test','feature_extraction_0', 'subgroup_1'):'X_test_data_01',
                    ('X_test','feature_extraction_1', 'subgroup_0'):'X_test_data_10',
                    ('X_test','feature_extraction_1', 'subgroup_1'):'X_test_data_11',
                },
                {
                    MATCH_KEY:[1,2],
                    ('y_test','voxel_extract_0','subgroup_0'):'y_test_data_00',
                    ('y_test','voxel_extract_0','subgroup_1'):'y_test_data_01',
                    ('y_test','voxel_extract_1','subgroup_0'):'y_test_data_10',
                    ('y_test','voxel_extract_1','subgroup_1'):'y_test_data_11',
                },
                {
                    MATCH_KEY:[1,2,4],
                    ('X_train','feature_extraction_0','subgroup_0',
                     'y_train','voxel_extract_0','RF'): 'RF_000',
                    ('X_train','feature_extraction_0','subgroup_0',
                     'y_train','voxel_extract_1','RF'): 'RF_001',
                    ('X_train','feature_extraction_0','subgroup_1',
                     'y_train','voxel_extract_0','RF'): 'RF_010',
                    ('X_train','feature_extraction_0','subgroup_1',
                     'y_train','voxel_extract_1','RF'): 'RF_011',
                    ('X_train','feature_extraction_1','subgroup_0',
                     'y_train','voxel_extract_0','RF'): 'RF_100',
                    ('X_train','feature_extraction_1','subgroup_0',
                     'y_train','voxel_extract_1','RF'): 'RF_101',
                    ('X_train','feature_extraction_1','subgroup_1',
                     'y_train','voxel_extract_0','RF'): 'RF_110',
                    ('X_train','feature_extraction_1','subgroup_1',
                     'y_train','voxel_extract_1','RF'): 'RF_111',
                },
            ],
            # out_dict
            {
                MATCH_KEY:[1,2,4],
                PREV_KEY:(),
                ('X_test','feature_extraction_0','subgroup_0',
                 'y_test','voxel_extract_0',
                 'X_train','y_train','RF'): ('X_test_data_00','y_test_data_00','RF_000'),
                ('X_test','feature_extraction_0','subgroup_0',
                 'y_test','voxel_extract_1',
                 'X_train','y_train','RF'): ('X_test_data_00','y_test_data_10','RF_001'),
                ('X_test','feature_extraction_0','subgroup_1',
                 'y_test','voxel_extract_0',
                 'X_train','y_train','RF'): ('X_test_data_01','y_test_data_01', 'RF_010'),
                ('X_test','feature_extraction_0','subgroup_1',
                 'y_test','voxel_extract_1',
                 'X_train','y_train','RF'): ('X_test_data_01','y_test_data_10', 'RF_011'),
                ('X_test','feature_extraction_1','subgroup_0',
                 'y_test','voxel_extract_0',
                 'X_train','y_train','RF'): ('X_test_data_10','y_test_data_00', 'RF_100'),
                ('X_test','feature_extraction_1','subgroup_0',
                 'y_test','voxel_extract_1',
                 'X_train','y_train','RF'): ('X_test_data_10','y_test_data_10', 'RF_101'),
                ('X_test','feature_extraction_1','subgroup_1',
                 'y_test','voxel_extract_0',
                 'X_train','y_train','RF'): ('X_test_data_11','y_test_data_01', 'RF_110'),
                ('X_test','feature_extraction_1','subgroup_1',
                 'y_test','voxel_extract_1',
                 'X_train','y_train','RF'): ('X_test_data_11','y_test_data_11', 'RF_111'),

            },
            marks=pytest.mark.xfail
        ),
        (
            [
                {
                    MATCH_KEY: [1],
                    ('X_train','standardize_0'):'X_train_0',
                    ('X_train','standardize_1'):'X_train_1',
                },
                {('y_train',):'y_train_data'}
            ],
            # out_dict
            {
                PREV_KEY:(), MATCH_KEY:[1],
                ('X_train','standardize_0','y_train'):('X_train_0','y_train_data'),
                ('X_train','standardize_1','y_train'):('X_train_1','y_train_data')
            }
        )
    ]
)
class TestCombineDicts:

    def test_combine_dicts(self, in_dicts, out_dict):
        result_dict = combine_dicts(*in_dicts)
        assert_equal(result_dict, out_dict, verbose=True)


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
                MATCH_KEY: [],
                ('data', 'module_0'): 5,
                ('data', 'module_1'): 6
            }
        ),
        (
            # in_dicts
            [
                # modules
                {
                    MATCH_KEY: [0],
                    ('group_0','module_0',): lambda x, y: x+y,
                    ('group_1','module_1',): lambda x, y: x*y,
                },
                # data_dict
                {
                    MATCH_KEY: [1],
                    ('data','group_0'): [np.array([1,2,3]), np.array([4,5,6])],
                    ('data','group_1'): [np.array([1,2,3]), np.array([4,5,6])],
                }
            ],
            # out_dict
            {
                MATCH_KEY: [1],
                ('data','group_0','module_0'): np.array([5,7,9]),
                ('data','group_1','module_1'): np.array([4,10,18]),
            }
        ),
        (
            # in_dicts
            [
                # modules
                {
                    MATCH_KEY: [1],
                    ('data','group_0','module_0',): lambda x, y: x+y,
                    ('data','group_1','module_1',): lambda x, y: x*y,
                },
                # data_dict
                {
                    MATCH_KEY: [1],
                    ('data','group_0'): [np.array([1,2,3]), np.array([4,5,6])],
                    ('data','group_1'): [np.array([1,2,3]), np.array([4,5,6])],
                }
            ],
            # out_dict
            {
                MATCH_KEY: [1],
                ('data','group_0','data','module_0'): np.array([5,7,9]),
                ('data','group_1','data','module_1'): np.array([4,10,18]),
            },
        ),
    ]
)
class TestApplyModules:

    def test_apply_modules(self, in_dicts, out_dict):
        result_dict = apply_modules(*in_dicts)
        assert_equal(result_dict, out_dict)
