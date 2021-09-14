import pytest

from vflow.module_set import PREV_KEY
from vflow.smart_subkey import SmartSubkey as sm
from vflow.convert import *

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
                PREV_KEY:('prev_0','prev_1',),
                ('X_train','y_train','RF','X_test'):('RF_fitted','X_test_data'),
                ('X_train','y_train','LR','X_test'):('LR_fitted','X_test_data')
            }
        ),
        (
            # in_dicts
            [
                {
                    PREV_KEY:('prev_0','prev_1',),
                    ('X_train','y_train','RF','X_test'):['RF_fitted','X_test_data'],
                    ('X_train','y_train','LR','X_test'):['LR_fitted','X_test_data']
                },
                {
                    ('y_test',):'y_test_data', ('y_test',):'y_test_data',
                    PREV_KEY:('prev_2',),
                }
            ],
            # out_dict
            {
                PREV_KEY:('prev_0','prev_1','prev_2',),
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
                PREV_KEY:()
            }
        ),
        (
            # in_dicts
            [
                {
                    ('X_train','y_train', sm('subsampling_0', 'origin_0'),'RF'):'RF_fitted_0',
                    ('X_train','y_train',sm('subsampling_1', 'origin_0'),'RF'):'RF_fitted_1',
                    ('X_train','y_train',sm('subsampling_0', 'origin_0'),'LR'):'LR_fitted_0',
                    ('X_train','y_train',sm('subsampling_1', 'origin_0'),'LR'):'LR_fitted_1',
                },
                {
                    ('X_train',sm('subsampling_0', 'origin_0')):'X_train_data_0',
                    ('X_train',sm('subsampling_1', 'origin_0')):'X_train_data_1',
                },
                {
                    ('y_train',sm('subsampling_0', 'origin_0')):'y_train_data_0',
                    ('y_train',sm('subsampling_1', 'origin_0')):'y_train_data_1',
                }
            ],
            # out_dict
            {
                ('X_train','y_train',sm('subsampling_0', 'origin_0'),'RF','X_train','y_train'):(
                    'RF_fitted_0','X_train_data_0','y_train_data_0'
                ),
                ('X_train','y_train',sm('subsampling_1', 'origin_0'),'RF','X_train','y_train'):(
                    'RF_fitted_1','X_train_data_1','y_train_data_1'
                ),
                ('X_train','y_train',sm('subsampling_0', 'origin_0'),'LR','X_train','y_train'):(
                    'LR_fitted_0','X_train_data_0','y_train_data_0'
                ),
                ('X_train','y_train',sm('subsampling_1', 'origin_0'),'LR','X_train','y_train'):(
                    'LR_fitted_1','X_train_data_1','y_train_data_1'
                ),
                PREV_KEY:()
            }
        ),
        (
            # in_dicts
            [
                {
                    ('X_train','y_train',sm('subgroup_0', 's_origin'),sm('voxel_extract_0', 'v_origin'),'RF'):'RF_fitted_00',
                    ('X_train','y_train',sm('subgroup_0', 's_origin'),sm('voxel_extract_1', 'v_origin'),'RF'):'RF_fitted_01',
                    ('X_train','y_train',sm('subgroup_1', 's_origin'),sm('voxel_extract_0', 'v_origin'),'RF'):'RF_fitted_10',
                    ('X_train','y_train',sm('subgroup_1', 's_origin'),sm('voxel_extract_1', 'v_origin'),'RF'):'RF_fitted_11',
                    ('X_train','y_train',sm('subgroup_0', 's_origin'),sm('voxel_extract_0', 'v_origin'),'LR'):'LR_fitted_00',
                    ('X_train','y_train',sm('subgroup_0', 's_origin'),sm('voxel_extract_1', 'v_origin'),'LR'):'LR_fitted_01',
                    ('X_train','y_train',sm('subgroup_1', 's_origin'),sm('voxel_extract_0', 'v_origin'),'LR'):'LR_fitted_10',
                    ('X_train','y_train',sm('subgroup_1', 's_origin'),sm('voxel_extract_1', 'v_origin'),'LR'):'LR_fitted_11'
                },
                {
                    ('X_test',sm('subgroup_0', 's_origin'),sm('voxel_extract_0', 'v_origin')):'X_test_data_00',
                    ('X_test',sm('subgroup_0', 's_origin'),sm('voxel_extract_1', 'v_origin')):'X_test_data_01',
                    ('X_test',sm('subgroup_1', 's_origin'),sm('voxel_extract_0', 'v_origin')):'X_test_data_10',
                    ('X_test',sm('subgroup_1', 's_origin'),sm('voxel_extract_1', 'v_origin')):'X_test_data_11'
                }
            ],
            # out_dict
            {
                ('X_train','y_train',sm('subgroup_0', 's_origin'),sm('voxel_extract_0', 'v_origin'),'RF','X_test'):(
                    'RF_fitted_00','X_test_data_00'
                ),
                ('X_train','y_train',sm('subgroup_0', 's_origin'),sm('voxel_extract_1', 'v_origin'),'RF','X_test'):(
                    'RF_fitted_01','X_test_data_01'
                ),
                ('X_train','y_train',sm('subgroup_1', 's_origin'),sm('voxel_extract_0', 'v_origin'),'RF','X_test'):(
                    'RF_fitted_10','X_test_data_10'
                ),
                ('X_train','y_train',sm('subgroup_1', 's_origin'),sm('voxel_extract_1', 'v_origin'),'RF','X_test'):(
                    'RF_fitted_11','X_test_data_11'
                ),
                ('X_train','y_train',sm('subgroup_0', 's_origin'),sm('voxel_extract_0', 'v_origin'),'LR','X_test'):(
                    'LR_fitted_00','X_test_data_00'
                ),
                ('X_train','y_train',sm('subgroup_0', 's_origin'),sm('voxel_extract_1', 'v_origin'),'LR','X_test'):(
                    'LR_fitted_01','X_test_data_01'
                ),
                ('X_train','y_train',sm('subgroup_1', 's_origin'),sm('voxel_extract_0', 'v_origin'),'LR','X_test'):(
                    'LR_fitted_10','X_test_data_10'
                ),
                ('X_train','y_train',sm('subgroup_1', 's_origin'),sm('voxel_extract_1', 'v_origin'),'LR','X_test'):(
                    'LR_fitted_11','X_test_data_11'
                ),
                PREV_KEY:()
            }
        ),
        (
            # in_dicts
            [
                {
                    ('X_train','y_train',sm('subgroup_0', 's_origin'),sm('voxel_extract_0', 'v_origin'),'RF'):'RF_fitted_00',
                    ('X_train','y_train',sm('subgroup_0', 's_origin'),sm('voxel_extract_1', 'v_origin'),'RF'):'RF_fitted_01',
                    ('X_train','y_train',sm('subgroup_1', 's_origin'),sm('voxel_extract_0', 'v_origin'),'RF'):'RF_fitted_10',
                    ('X_train','y_train',sm('subgroup_1', 's_origin'),sm('voxel_extract_1', 'v_origin'),'RF'):'RF_fitted_11',
                    ('X_train','y_train',sm('subgroup_0', 's_origin'),sm('voxel_extract_0', 'v_origin'),'LR'):'LR_fitted_00',
                    ('X_train','y_train',sm('subgroup_0', 's_origin'),sm('voxel_extract_1', 'v_origin'),'LR'):'LR_fitted_01',
                    ('X_train','y_train',sm('subgroup_1', 's_origin'),sm('voxel_extract_0', 'v_origin'),'LR'):'LR_fitted_10',
                    ('X_train','y_train',sm('subgroup_1', 's_origin'),sm('voxel_extract_1', 'v_origin'),'LR'):'LR_fitted_11'
                },
                {
                    ('X_test',sm('subgroup_0', 's_origin'),sm('voxel_extract_0', 'v_origin')):'X_test_data_00',
                    ('X_test',sm('subgroup_0', 's_origin'),sm('voxel_extract_1', 'v_origin')):'X_test_data_01',
                    ('X_test',sm('subgroup_1', 's_origin'),sm('voxel_extract_0', 'v_origin')):'X_test_data_10',
                    ('X_test',sm('subgroup_1', 's_origin'),sm('voxel_extract_1', 'v_origin')):'X_test_data_11'
                }
            ],
            # out_dict
            {
                ('X_train','y_train',sm('subgroup_0', 's_origin'),sm('voxel_extract_0', 'v_origin'),'RF','X_test'):(
                    'RF_fitted_00','X_test_data_00'
                ),
                ('X_train','y_train',sm('subgroup_0', 's_origin'),sm('voxel_extract_1', 'v_origin'),'RF','X_test'):(
                    'RF_fitted_01','X_test_data_01'
                ),
                ('X_train','y_train',sm('subgroup_1', 's_origin'),sm('voxel_extract_0', 'v_origin'),'RF','X_test'):(
                    'RF_fitted_10','X_test_data_10'
                ),
                ('X_train','y_train',sm('subgroup_1', 's_origin'),sm('voxel_extract_1', 'v_origin'),'RF','X_test'):(
                    'RF_fitted_11','X_test_data_11'
                ),
                ('X_train','y_train',sm('subgroup_0', 's_origin'),sm('voxel_extract_0', 'v_origin'),'LR','X_test'):(
                    'LR_fitted_00','X_test_data_00'
                ),
                ('X_train','y_train',sm('subgroup_0', 's_origin'),sm('voxel_extract_1', 'v_origin'),'LR','X_test'):(
                    'LR_fitted_01','X_test_data_01'
                ),
                ('X_train','y_train',sm('subgroup_1', 's_origin'),sm('voxel_extract_0', 'v_origin'),'LR','X_test'):(
                    'LR_fitted_10','X_test_data_10'
                ),
                ('X_train','y_train',sm('subgroup_1', 's_origin'),sm('voxel_extract_1', 'v_origin'),'LR','X_test'):(
                    'LR_fitted_11','X_test_data_11'
                ),
                PREV_KEY:()
            }
        ),
        (
            # in_dicts
            [
                {
                    ('X_train','y_train',sm('subgroup_0', 's_origin'),sm('voxel_extract_0', 'v_origin'),'RF','X_test'):[
                        'RF_fitted_00','X_test_data_00'
                    ],
                    ('X_train','y_train',sm('subgroup_0', 's_origin'),sm('voxel_extract_1', 'v_origin'),'RF','X_test'):[
                        'RF_fitted_01','X_test_data_01'
                    ],
                    ('X_train','y_train',sm('subgroup_1', 's_origin'),sm('voxel_extract_0', 'v_origin'),'RF','X_test'):[
                        'RF_fitted_10','X_test_data_10'
                    ],
                    ('X_train','y_train',sm('subgroup_1', 's_origin'),sm('voxel_extract_1', 'v_origin'),'RF','X_test'):[
                        'RF_fitted_11','X_test_data_11'
                    ],
                    ('X_train','y_train',sm('subgroup_0', 's_origin'),sm('voxel_extract_0', 'v_origin'),'LR','X_test'):[
                        'LR_fitted_00','X_test_data_00'
                    ],
                    ('X_train','y_train',sm('subgroup_0', 's_origin'),sm('voxel_extract_1', 'v_origin'),'LR','X_test'):[
                        'LR_fitted_01','X_test_data_01'
                    ],
                    ('X_train','y_train',sm('subgroup_1', 's_origin'),sm('voxel_extract_0', 'v_origin'),'LR','X_test'):[
                        'LR_fitted_10','X_test_data_10'
                    ],
                    ('X_train','y_train',sm('subgroup_1', 's_origin'),sm('voxel_extract_1', 'v_origin'),'LR','X_test'):[
                        'LR_fitted_11','X_test_data_11'
                    ],
                    PREV_KEY:()
                },
                {
                    ('y_test',sm('subgroup_0', 's_origin'),sm('voxel_extract_0', 'v_origin')):'y_test_data_00',
                    ('y_test',sm('subgroup_0', 's_origin'),sm('voxel_extract_1', 'v_origin')):'y_test_data_01',
                    ('y_test',sm('subgroup_1', 's_origin'),sm('voxel_extract_0', 'v_origin')):'y_test_data_10',
                    ('y_test',sm('subgroup_1', 's_origin'),sm('voxel_extract_1', 'v_origin')):'y_test_data_11'
                }
            ],
            # out_dict
            {
                ('X_train','y_train',sm('subgroup_0', 's_origin'),sm('voxel_extract_0', 'v_origin'),'RF','X_test','y_test'):(
                    ['RF_fitted_00','X_test_data_00'],'y_test_data_00'
                ),
                ('X_train','y_train',sm('subgroup_0', 's_origin'),sm('voxel_extract_1', 'v_origin'),'RF','X_test','y_test'):(
                    ['RF_fitted_01','X_test_data_01'],'y_test_data_01'
                ),
                ('X_train','y_train',sm('subgroup_1', 's_origin'),sm('voxel_extract_0', 'v_origin'),'RF','X_test','y_test'):(
                    ['RF_fitted_10','X_test_data_10'],'y_test_data_10'
                ),
                ('X_train','y_train',sm('subgroup_1', 's_origin'),sm('voxel_extract_1', 'v_origin'),'RF','X_test','y_test'):(
                    ['RF_fitted_11','X_test_data_11'],'y_test_data_11'
                ),
                ('X_train','y_train',sm('subgroup_0', 's_origin'),sm('voxel_extract_0', 'v_origin'),'LR','X_test','y_test'):(
                    ['LR_fitted_00','X_test_data_00'],'y_test_data_00'
                ),
                ('X_train','y_train',sm('subgroup_0', 's_origin'),sm('voxel_extract_1', 'v_origin'),'LR','X_test','y_test'):(
                    ['LR_fitted_01','X_test_data_01'],'y_test_data_01'
                ),
                ('X_train','y_train',sm('subgroup_1', 's_origin'),sm('voxel_extract_0', 'v_origin'),'LR','X_test','y_test'):(
                    ['LR_fitted_10','X_test_data_10'],'y_test_data_10'
                ),
                ('X_train','y_train',sm('subgroup_1', 's_origin'),sm('voxel_extract_1', 'v_origin'),'LR','X_test','y_test'):(
                    ['LR_fitted_11','X_test_data_11'],'y_test_data_11'
                ),
                PREV_KEY:()
            }
        ),
        (
            # in_dicts
            [
                {
                    ('X_train','y_train',sm('subgroup_0', 's_origin'),sm('voxel_extract_0', 'v_origin'),sm('RF', 'm_origin'),'X_test','y_test'):[
                        ['RF_fitted_00','X_test_data_00'],'y_test_data'
                    ],
                    ('X_train','y_train',sm('subgroup_0', 's_origin'),sm('voxel_extract_1', 'v_origin'),sm('RF', 'm_origin'),'X_test','y_test'):[
                        ['RF_fitted_01','X_test_data_01'],'y_test_data'
                    ],
                    ('X_train','y_train',sm('subgroup_1', 's_origin'),sm('voxel_extract_0', 'v_origin'),sm('RF', 'm_origin'),'X_test','y_test'):[
                        ['RF_fitted_10','X_test_data_10'],'y_test_data'
                    ],
                    ('X_train','y_train',sm('subgroup_1', 's_origin'),sm('voxel_extract_1', 'v_origin'),sm('RF', 'm_origin'),'X_test','y_test'):[
                        ['RF_fitted_11','X_test_data_11'],'y_test_data'
                    ],
                    ('X_train','y_train',sm('subgroup_0', 's_origin'),sm('voxel_extract_0', 'v_origin'),sm('LR', 'm_origin'),'X_test','y_test'):[
                        ['LR_fitted_00','X_test_data_00'],'y_test_data'
                    ],
                    ('X_train','y_train',sm('subgroup_0', 's_origin'),sm('voxel_extract_1', 'v_origin'),sm('LR', 'm_origin'),'X_test','y_test'):[
                        ['LR_fitted_01','X_test_data_01'],'y_test_data'
                    ],
                    ('X_train','y_train',sm('subgroup_1', 's_origin'),sm('voxel_extract_0', 'v_origin'),sm('LR', 'm_origin'),'X_test','y_test'):[
                        ['LR_fitted_10','X_test_data_10'],'y_test_data'
                    ],
                    ('X_train','y_train',sm('subgroup_1', 's_origin'),sm('voxel_extract_1', 'v_origin'),sm('LR', 'm_origin'),'X_test','y_test'):[
                        ['LR_fitted_11','X_test_data_11'],'y_test_data'
                    ],
                    PREV_KEY:()
                },
                {
                    (sm('LR', 'm_origin'),'acc'):'LR_acc_func',
                    (sm('LR', 'm_origin'),'bal_acc'):'LR_bal_acc_func',
                    (sm('RF', 'm_origin'),'acc'):'RF_acc_func',
                    (sm('RF', 'm_origin'),'feat_imp'):'RF_feat_imp_func'
                }
            ],
            # out_dict
            {
                ('X_train','y_train',sm('subgroup_0', 's_origin'),sm('voxel_extract_0', 'v_origin'),sm('RF', 'm_origin'),'X_test','y_test','acc'):(
                    [['RF_fitted_00','X_test_data_00'],'y_test_data'],'RF_acc_func'
                ),
                ('X_train','y_train',sm('subgroup_0', 's_origin'),sm('voxel_extract_0', 'v_origin'),sm('RF', 'm_origin'),'X_test','y_test','feat_imp'):(
                    [['RF_fitted_00','X_test_data_00'],'y_test_data'],'RF_feat_imp_func'
                ),
                ('X_train','y_train',sm('subgroup_0', 's_origin'),sm('voxel_extract_1', 'v_origin'),sm('RF', 'm_origin'),'X_test','y_test','acc'):(
                    [['RF_fitted_01','X_test_data_01'],'y_test_data'],'RF_acc_func'
                ),
                ('X_train','y_train',sm('subgroup_0', 's_origin'),sm('voxel_extract_1', 'v_origin'),sm('RF', 'm_origin'),'X_test','y_test','feat_imp'):(
                    [['RF_fitted_01','X_test_data_01'],'y_test_data'],'RF_feat_imp_func'
                ),
                ('X_train','y_train',sm('subgroup_1', 's_origin'),sm('voxel_extract_0', 'v_origin'),sm('RF', 'm_origin'),'X_test','y_test','acc'):(
                    [['RF_fitted_10','X_test_data_10'],'y_test_data'],'RF_acc_func'
                ),
                ('X_train','y_train',sm('subgroup_1', 's_origin'),sm('voxel_extract_0', 'v_origin'),sm('RF', 'm_origin'),'X_test','y_test','feat_imp'):(
                    [['RF_fitted_10','X_test_data_10'],'y_test_data'],'RF_feat_imp_func'
                ),
                ('X_train','y_train',sm('subgroup_1', 's_origin'),sm('voxel_extract_1', 'v_origin'),sm('RF', 'm_origin'),'X_test','y_test','acc'):(
                    [['RF_fitted_11','X_test_data_11'],'y_test_data'],'RF_acc_func'
                ),
                ('X_train','y_train',sm('subgroup_1', 's_origin'),sm('voxel_extract_1', 'v_origin'),sm('RF', 'm_origin'),'X_test','y_test','feat_imp'):(
                    [['RF_fitted_11','X_test_data_11'],'y_test_data'],'RF_feat_imp_func'
                ),
                ('X_train','y_train',sm('subgroup_0', 's_origin'),sm('voxel_extract_0', 'v_origin'),sm('LR', 'm_origin'),'X_test','y_test','acc'):(
                    [['LR_fitted_00','X_test_data_00'],'y_test_data'],'LR_acc_func'
                ),
                ('X_train','y_train',sm('subgroup_0', 's_origin'),sm('voxel_extract_0', 'v_origin'),sm('LR', 'm_origin'),'X_test','y_test','bal_acc'):(
                    [['LR_fitted_00','X_test_data_00'],'y_test_data'],'LR_bal_acc_func'
                ),
                ('X_train','y_train',sm('subgroup_0', 's_origin'),sm('voxel_extract_1', 'v_origin'),sm('LR', 'm_origin'),'X_test','y_test','acc'):(
                    [['LR_fitted_01','X_test_data_01'],'y_test_data'],'LR_acc_func'
                ),
                ('X_train','y_train',sm('subgroup_0', 's_origin'),sm('voxel_extract_1', 'v_origin'),sm('LR', 'm_origin'),'X_test','y_test','bal_acc'):(
                    [['LR_fitted_01','X_test_data_01'],'y_test_data'],'LR_bal_acc_func'
                ),
                ('X_train','y_train',sm('subgroup_1', 's_origin'),sm('voxel_extract_0', 'v_origin'),sm('LR', 'm_origin'),'X_test','y_test','acc'):(
                    [['LR_fitted_10','X_test_data_10'],'y_test_data'],'LR_acc_func'
                ),
                ('X_train','y_train',sm('subgroup_1', 's_origin'),sm('voxel_extract_0', 'v_origin'),sm('LR', 'm_origin'),'X_test','y_test','bal_acc'):(
                    [['LR_fitted_10','X_test_data_10'],'y_test_data'],'LR_bal_acc_func'
                ),
                ('X_train','y_train',sm('subgroup_1', 's_origin'),sm('voxel_extract_1', 'v_origin'),sm('LR', 'm_origin'),'X_test','y_test','acc'):(
                    [['LR_fitted_11','X_test_data_11'],'y_test_data'],'LR_acc_func'
                ),
                ('X_train','y_train',sm('subgroup_1', 's_origin'),sm('voxel_extract_1', 'v_origin'),sm('LR', 'm_origin'),'X_test','y_test','bal_acc'):(
                    [['LR_fitted_11','X_test_data_11'],'y_test_data'],'LR_bal_acc_func'
                ),
                PREV_KEY:()
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
                    ('X_train','y_train',sm('voxel_extract_0', 'v_origin'),sm('subgroup_0', 's_origin'), 'RF'):'RF_00',
                    ('X_train','y_train',sm('voxel_extract_0', 'v_origin'),sm('subgroup_1', 's_origin'), 'RF'):'RF_01',
                    ('X_train','y_train',sm('voxel_extract_1', 'v_origin'),sm('subgroup_0', 's_origin'), 'RF'):'RF_10',
                    ('X_train','y_train',sm('voxel_extract_1', 'v_origin'),sm('subgroup_1', 's_origin'), 'RF'):'RF_11',
                },
                {
                    ('X_test',sm('subgroup_0', 's_origin')):'X_test_data_0',
                    ('X_test',sm('subgroup_1', 's_origin')):'X_test_data_1'
                },
                {
                    ('y_test',sm('voxel_extract_0', 'v_origin'),sm('subgroup_0', 's_origin')):'y_test_data_00',
                    ('y_test',sm('voxel_extract_0', 'v_origin'),sm('subgroup_1', 's_origin')):'y_test_data_01',
                    ('y_test',sm('voxel_extract_1', 'v_origin'),sm('subgroup_0', 's_origin')):'y_test_data_10',
                    ('y_test',sm('voxel_extract_1', 'v_origin'),sm('subgroup_1', 's_origin')):'y_test_data_11',
                }
            ],
            # out_dict
            {
                PREV_KEY:(),
                ('X_train','y_train',sm('voxel_extract_0', 'v_origin'),sm('subgroup_0', 's_origin'),'RF','X_test','y_test'):(
                    'RF_00', 'X_test_data_0', 'y_test_data_00'
                ),
                ('X_train','y_train',sm('voxel_extract_0', 'v_origin'),sm('subgroup_1', 's_origin'),'RF','X_test','y_test'):(
                    'RF_01', 'X_test_data_1', 'y_test_data_01'
                ),
                ('X_train','y_train',sm('voxel_extract_1', 'v_origin'),sm('subgroup_0', 's_origin'),'RF','X_test','y_test'):(
                    'RF_10', 'X_test_data_0', 'y_test_data_10'
                ),
                ('X_train','y_train',sm('voxel_extract_1', 'v_origin'),sm('subgroup_1', 's_origin'),'RF','X_test','y_test'):(
                    'RF_11', 'X_test_data_1', 'y_test_data_11'
                )
            },
        ),
        (
            # in_dicts
            [
                {
                    ('X_train','y_train',sm('voxel_extract_0', 'v_origin'),sm('subgroup_0', 's_origin'), 'RF'):'RF_00',
                    ('X_train','y_train',sm('voxel_extract_0', 'v_origin'),sm('subgroup_1', 's_origin'), 'RF'):'RF_01',
                    ('X_train','y_train',sm('voxel_extract_1', 'v_origin'),sm('subgroup_0', 's_origin'), 'RF'):'RF_10',
                    ('X_train','y_train',sm('voxel_extract_1', 'v_origin'),sm('subgroup_1', 's_origin'), 'RF'):'RF_11',
                },
                {
                    ('y_test',sm('voxel_extract_0', 'v_origin'),sm('subgroup_0', 's_origin')):'y_test_data_00',
                    ('y_test',sm('voxel_extract_0', 'v_origin'),sm('subgroup_1', 's_origin')):'y_test_data_01',
                    ('y_test',sm('voxel_extract_1', 'v_origin'),sm('subgroup_0', 's_origin')):'y_test_data_10',
                    ('y_test',sm('voxel_extract_1', 'v_origin'),sm('subgroup_1', 's_origin')):'y_test_data_11',
                },
                {
                    ('X_test',sm('subgroup_0', 's_origin')):'X_test_data_0',
                    ('X_test',sm('subgroup_1', 's_origin')):'X_test_data_1'
                },
            ],
            # out_dict
            {
                PREV_KEY:(),
                ('X_train','y_train',sm('voxel_extract_0', 'v_origin'),sm('subgroup_0', 's_origin'),'RF','y_test','X_test'):(
                    'RF_00', 'y_test_data_00', 'X_test_data_0'
                ),
                ('X_train','y_train',sm('voxel_extract_0', 'v_origin'),sm('subgroup_1', 's_origin'),'RF','y_test','X_test'):(
                    'RF_01', 'y_test_data_01', 'X_test_data_1'
                ),
                ('X_train','y_train',sm('voxel_extract_1', 'v_origin'),sm('subgroup_0', 's_origin'),'RF','y_test','X_test'):(
                    'RF_10', 'y_test_data_10', 'X_test_data_0'
                ),
                ('X_train','y_train',sm('voxel_extract_1', 'v_origin'),sm('subgroup_1', 's_origin'),'RF','y_test','X_test'):(
                    'RF_11', 'y_test_data_11', 'X_test_data_1'
                )
            },
        ),
        (
            # in_dicts
            [
                {
                    ('X_test',sm('subgroup_0', 's_origin')):'X_test_data_0',
                    ('X_test',sm('subgroup_1', 's_origin')):'X_test_data_1'
                },
                {
                    ('X_train','y_train',sm('voxel_extract_0', 'v_origin'),sm('subgroup_0', 's_origin'), 'RF'):'RF_00',
                    ('X_train','y_train',sm('voxel_extract_0', 'v_origin'),sm('subgroup_1', 's_origin'), 'RF'):'RF_01',
                    ('X_train','y_train',sm('voxel_extract_1', 'v_origin'),sm('subgroup_0', 's_origin'), 'RF'):'RF_10',
                    ('X_train','y_train',sm('voxel_extract_1', 'v_origin'),sm('subgroup_1', 's_origin'), 'RF'):'RF_11',
                },
                {
                    ('y_test',sm('voxel_extract_0', 'v_origin'),sm('subgroup_0', 's_origin')):'y_test_data_00',
                    ('y_test',sm('voxel_extract_0', 'v_origin'),sm('subgroup_1', 's_origin')):'y_test_data_01',
                    ('y_test',sm('voxel_extract_1', 'v_origin'),sm('subgroup_0', 's_origin')):'y_test_data_10',
                    ('y_test',sm('voxel_extract_1', 'v_origin'),sm('subgroup_1', 's_origin')):'y_test_data_11',
                }
            ],
            # out_dict
            {
                PREV_KEY:(),
                ('X_test',sm('subgroup_0', 's_origin'),'X_train','y_train',sm('voxel_extract_0', 'v_origin'),'RF','y_test'):(
                    'X_test_data_0', 'RF_00', 'y_test_data_00'
                ),
                ('X_test',sm('subgroup_1', 's_origin'),'X_train','y_train',sm('voxel_extract_0', 'v_origin'),'RF','y_test'):(
                    'X_test_data_1', 'RF_01', 'y_test_data_01'
                ),
                ('X_test',sm('subgroup_0', 's_origin'),'X_train','y_train',sm('voxel_extract_1', 'v_origin'),'RF','y_test'):(
                    'X_test_data_0', 'RF_10', 'y_test_data_10'
                ),
                ('X_test',sm('subgroup_1', 's_origin'),'X_train','y_train',sm('voxel_extract_1', 'v_origin'),'RF','y_test'):(
                    'X_test_data_1', 'RF_11', 'y_test_data_11'
                )
            },
        ),
        (
            # in_dicts
            [
                {
                    ('X_test',sm('subgroup_0', 's_origin')):'X_test_data_0',
                    ('X_test',sm('subgroup_1', 's_origin')):'X_test_data_1'
                },
                {
                    ('y_test',sm('voxel_extract_0', 'v_origin'),sm('subgroup_0', 's_origin')):'y_test_data_00',
                    ('y_test',sm('voxel_extract_0', 'v_origin'),sm('subgroup_1', 's_origin')):'y_test_data_01',
                    ('y_test',sm('voxel_extract_1', 'v_origin'),sm('subgroup_0', 's_origin')):'y_test_data_10',
                    ('y_test',sm('voxel_extract_1', 'v_origin'),sm('subgroup_1', 's_origin')):'y_test_data_11',
                },
                {
                    ('X_train','y_train',sm('voxel_extract_0', 'v_origin'),sm('subgroup_0', 's_origin'), 'RF'):'RF_00',
                    ('X_train','y_train',sm('voxel_extract_0', 'v_origin'),sm('subgroup_1', 's_origin'), 'RF'):'RF_01',
                    ('X_train','y_train',sm('voxel_extract_1', 'v_origin'),sm('subgroup_0', 's_origin'), 'RF'):'RF_10',
                    ('X_train','y_train',sm('voxel_extract_1', 'v_origin'),sm('subgroup_1', 's_origin'), 'RF'):'RF_11',
                },
            ],
            # out_dict
            {
                PREV_KEY:(),
                ('X_test',sm('subgroup_0', 's_origin'),'y_test',sm('voxel_extract_0', 'v_origin'),'X_train','y_train','RF'):(
                    'X_test_data_0', 'y_test_data_00', 'RF_00'
                ),
                ('X_test',sm('subgroup_1', 's_origin'),'y_test',sm('voxel_extract_0', 'v_origin'),'X_train','y_train','RF'):(
                    'X_test_data_1', 'y_test_data_01', 'RF_01'
                ),
                ('X_test',sm('subgroup_0', 's_origin'),'y_test',sm('voxel_extract_1', 'v_origin'),'X_train','y_train','RF'):(
                    'X_test_data_0', 'y_test_data_10', 'RF_10'
                ),
                ('X_test',sm('subgroup_1', 's_origin'),'y_test',sm('voxel_extract_1', 'v_origin'),'X_train','y_train','RF'):(
                    'X_test_data_1', 'y_test_data_11', 'RF_11'
                )
            },
        ),
        (
            # in_dicts
            [
                {
                    ('y_test',sm('voxel_extract_0', 'v_origin'),sm('subgroup_0', 's_origin')):'y_test_data_00',
                    ('y_test',sm('voxel_extract_0', 'v_origin'),sm('subgroup_1', 's_origin')):'y_test_data_01',
                    ('y_test',sm('voxel_extract_1', 'v_origin'),sm('subgroup_0', 's_origin')):'y_test_data_10',
                    ('y_test',sm('voxel_extract_1', 'v_origin'),sm('subgroup_1', 's_origin')):'y_test_data_11',
                },
                {
                    ('X_train','y_train',sm('voxel_extract_0', 'v_origin'),sm('subgroup_0', 's_origin'), 'RF'):'RF_00',
                    ('X_train','y_train',sm('voxel_extract_0', 'v_origin'),sm('subgroup_1', 's_origin'), 'RF'):'RF_01',
                    ('X_train','y_train',sm('voxel_extract_1', 'v_origin'),sm('subgroup_0', 's_origin'), 'RF'):'RF_10',
                    ('X_train','y_train',sm('voxel_extract_1', 'v_origin'),sm('subgroup_1', 's_origin'), 'RF'):'RF_11',
                },
                {
                    ('X_test',sm('subgroup_0', 's_origin')):'X_test_data_0',
                    ('X_test',sm('subgroup_1', 's_origin')):'X_test_data_1'
                },
            ],
            # out_dict
            {
                PREV_KEY:(),
                ('y_test',sm('voxel_extract_0', 'v_origin'),sm('subgroup_0', 's_origin'),'X_train','y_train','RF','X_test'):(
                    'y_test_data_00', 'RF_00', 'X_test_data_0'
                ),
                ('y_test',sm('voxel_extract_0', 'v_origin'),sm('subgroup_1', 's_origin'),'X_train','y_train','RF','X_test'):(
                    'y_test_data_01', 'RF_01', 'X_test_data_1'
                ),
                ('y_test',sm('voxel_extract_1', 'v_origin'),sm('subgroup_0', 's_origin'),'X_train','y_train','RF','X_test'):(
                    'y_test_data_10', 'RF_10', 'X_test_data_0'
                ),
                ('y_test',sm('voxel_extract_1', 'v_origin'),sm('subgroup_1', 's_origin'),'X_train','y_train','RF','X_test'):(
                    'y_test_data_11', 'RF_11', 'X_test_data_1'
                )
            },
        ),
        (
            # in_dicts
            [
                {
                    ('y_test',sm('voxel_extract_0', 'v_origin'),sm('subgroup_0', 's_origin')):'y_test_data_00',
                    ('y_test',sm('voxel_extract_0', 'v_origin'),sm('subgroup_1', 's_origin')):'y_test_data_01',
                    ('y_test',sm('voxel_extract_1', 'v_origin'),sm('subgroup_0', 's_origin')):'y_test_data_10',
                    ('y_test',sm('voxel_extract_1', 'v_origin'),sm('subgroup_1', 's_origin')):'y_test_data_11',
                },
                {
                    ('X_test',sm('subgroup_0', 's_origin')):'X_test_data_0',
                    ('X_test',sm('subgroup_1', 's_origin')):'X_test_data_1'
                },
                {
                    ('X_train','y_train',sm('voxel_extract_0', 'v_origin'),sm('subgroup_0', 's_origin'), 'RF'):'RF_00',
                    ('X_train','y_train',sm('voxel_extract_0', 'v_origin'),sm('subgroup_1', 's_origin'), 'RF'):'RF_01',
                    ('X_train','y_train',sm('voxel_extract_1', 'v_origin'),sm('subgroup_0', 's_origin'), 'RF'):'RF_10',
                    ('X_train','y_train',sm('voxel_extract_1', 'v_origin'),sm('subgroup_1', 's_origin'), 'RF'):'RF_11',
                },
            ],
            # out_dict
            {
                PREV_KEY:(),
                ('y_test',sm('voxel_extract_0', 'v_origin'),sm('subgroup_0', 's_origin'),'X_test','X_train','y_train','RF'):(
                    'y_test_data_00', 'X_test_data_0', 'RF_00'
                ),
                ('y_test',sm('voxel_extract_0', 'v_origin'),sm('subgroup_1', 's_origin'),'X_test','X_train','y_train','RF'):(
                    'y_test_data_01', 'X_test_data_1', 'RF_01'
                ),
                ('y_test',sm('voxel_extract_1', 'v_origin'),sm('subgroup_0', 's_origin'),'X_test','X_train','y_train','RF'):(
                    'y_test_data_10', 'X_test_data_0', 'RF_10'
                ),
                ('y_test',sm('voxel_extract_1', 'v_origin'),sm('subgroup_1', 's_origin'),'X_test','X_train','y_train','RF'):(
                    'y_test_data_11', 'X_test_data_1', 'RF_11'
                )
            },
        ),
        (
            # in_dicts
            [
                {
                    ('X_test',sm('feature_extraction_0', 'f_origin'), sm('subgroup_0', 's_origin')):'X_test_data_00',
                    ('X_test',sm('feature_extraction_0', 'f_origin'), sm('subgroup_1', 's_origin')):'X_test_data_01',
                    ('X_test',sm('feature_extraction_1', 'f_origin'), sm('subgroup_0', 's_origin')):'X_test_data_10',
                    ('X_test',sm('feature_extraction_1', 'f_origin'), sm('subgroup_1', 's_origin')):'X_test_data_11',
                },
                {
                    ('y_test',sm('voxel_extract_0', 'v_origin'),sm('subgroup_0', 's_origin')):'y_test_data_00',
                    ('y_test',sm('voxel_extract_0', 'v_origin'),sm('subgroup_1', 's_origin')):'y_test_data_01',
                    ('y_test',sm('voxel_extract_1', 'v_origin'),sm('subgroup_0', 's_origin')):'y_test_data_10',
                    ('y_test',sm('voxel_extract_1', 'v_origin'),sm('subgroup_1', 's_origin')):'y_test_data_11',
                },
                {
                    ('X_train',sm('feature_extraction_0', 'f_origin'),sm('subgroup_0', 's_origin'),
                     'y_train',sm('voxel_extract_0', 'v_origin'),'RF'): 'RF_000',
                    ('X_train',sm('feature_extraction_0', 'f_origin'),sm('subgroup_0', 's_origin'),
                     'y_train',sm('voxel_extract_1', 'v_origin'),'RF'): 'RF_001',
                    ('X_train',sm('feature_extraction_0', 'f_origin'),sm('subgroup_1', 's_origin'),
                     'y_train',sm('voxel_extract_0', 'v_origin'),'RF'): 'RF_010',
                    ('X_train',sm('feature_extraction_0', 'f_origin'),sm('subgroup_1', 's_origin'),
                     'y_train',sm('voxel_extract_1', 'v_origin'),'RF'): 'RF_011',
                    ('X_train',sm('feature_extraction_1', 'f_origin'),sm('subgroup_0', 's_origin'),
                     'y_train',sm('voxel_extract_0', 'v_origin'),'RF'): 'RF_100',
                    ('X_train',sm('feature_extraction_1', 'f_origin'),sm('subgroup_0', 's_origin'),
                     'y_train',sm('voxel_extract_1', 'v_origin'),'RF'): 'RF_101',
                    ('X_train',sm('feature_extraction_1', 'f_origin'),sm('subgroup_1', 's_origin'),
                     'y_train',sm('voxel_extract_0', 'v_origin'),'RF'): 'RF_110',
                    ('X_train',sm('feature_extraction_1', 'f_origin'),sm('subgroup_1', 's_origin'),
                     'y_train',sm('voxel_extract_1', 'v_origin'),'RF'): 'RF_111',
                },
            ],
            # out_dict
            {
                PREV_KEY:(),
                ('X_test',sm('feature_extraction_0', 'f_origin'),sm('subgroup_0', 's_origin'),
                 'y_test',sm('voxel_extract_0', 'v_origin'),
                 'X_train','y_train','RF'): ('X_test_data_00','y_test_data_00','RF_000'),
                ('X_test',sm('feature_extraction_0', 'f_origin'),sm('subgroup_0', 's_origin'),
                 'y_test',sm('voxel_extract_1', 'v_origin'),
                 'X_train','y_train','RF'): ('X_test_data_00','y_test_data_10','RF_001'),
                ('X_test',sm('feature_extraction_0', 'f_origin'),sm('subgroup_1', 's_origin'),
                 'y_test',sm('voxel_extract_0', 'v_origin'),
                 'X_train','y_train','RF'): ('X_test_data_01','y_test_data_01', 'RF_010'),
                ('X_test',sm('feature_extraction_0', 'f_origin'),sm('subgroup_1', 's_origin'),
                 'y_test',sm('voxel_extract_1', 'v_origin'),
                 'X_train','y_train','RF'): ('X_test_data_01','y_test_data_11', 'RF_011'),
                ('X_test',sm('feature_extraction_1', 'f_origin'),sm('subgroup_0', 's_origin'),
                 'y_test',sm('voxel_extract_0', 'v_origin'),
                 'X_train','y_train','RF'): ('X_test_data_10','y_test_data_00', 'RF_100'),
                ('X_test',sm('feature_extraction_1', 'f_origin'),sm('subgroup_0', 's_origin'),
                 'y_test',sm('voxel_extract_1', 'v_origin'),
                 'X_train','y_train','RF'): ('X_test_data_10','y_test_data_10', 'RF_101'),
                ('X_test',sm('feature_extraction_1', 'f_origin'),sm('subgroup_1', 's_origin'),
                 'y_test',sm('voxel_extract_0', 'v_origin'),
                 'X_train','y_train','RF'): ('X_test_data_11','y_test_data_01', 'RF_110'),
                ('X_test',sm('feature_extraction_1', 'f_origin'),sm('subgroup_1', 's_origin'),
                 'y_test',sm('voxel_extract_1', 'v_origin'),
                 'X_train','y_train','RF'): ('X_test_data_11','y_test_data_11', 'RF_111'),

            }
        ),
        (
            [
                {
                    ('X_train',sm('standardize_0', 's_origin')):'X_train_0',
                    ('X_train',sm('standardize_1', 's_origin')):'X_train_1',
                },
                {('y_train',):'y_train_data'}
            ],
            # out_dict
            {
                PREV_KEY:(),
                ('X_train',sm('standardize_0', 's_origin'),'y_train'):('X_train_0','y_train_data'),
                ('X_train',sm('standardize_1', 's_origin'),'y_train'):('X_train_1','y_train_data')
            }
        ),
    ]
)
class TestCombineDicts:

    def test_combine_dicts(self, in_dicts, out_dict):
        result_dict = combine_dicts(*in_dicts)
        print(result_dict)
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
                ('data', 'module_0'): 5,
                ('data', 'module_1'): 6
            }
        ),
        (
            # in_dicts
            [
                # modules
                {
                    (sm('group_0', 'g_origin'),'module_0',): lambda x, y: x+y,
                    (sm('group_1', 'g_origin'),'module_1',): lambda x, y: x*y,
                },
                # data_dict
                {
                    ('data',sm('group_0', 'g_origin')): [np.array([1,2,3]), np.array([4,5,6])],
                    ('data',sm('group_1', 'g_origin')): [np.array([1,2,3]), np.array([4,5,6])],
                }
            ],
            # out_dict
            {
                ('data',sm('group_0', 'g_origin'),'module_0'): np.array([5,7,9]),
                ('data',sm('group_1', 'g_origin'),'module_1'): np.array([4,10,18]),
            }
        ),
        (
            # in_dicts
            [
                # modules
                {
                    ('data',sm('group_0', 'g_origin'),'module_0',): lambda x, y: x+y,
                    ('data',sm('group_1', 'g_origin'),'module_1',): lambda x, y: x*y,
                },
                # data_dict
                {
                    ('data',sm('group_0', 'g_origin')): [np.array([1,2,3]), np.array([4,5,6])],
                    ('data',sm('group_1', 'g_origin')): [np.array([1,2,3]), np.array([4,5,6])],
                }
            ],
            # out_dict
            {
                ('data',sm('group_0', 'g_origin'),'data','module_0'): np.array([5,7,9]),
                ('data',sm('group_1', 'g_origin'),'data','module_1'): np.array([4,10,18]),
            },
        ),
    ]
)
class TestApplyModules:

    def test_apply_modules(self, in_dicts, out_dict):
        result_dict = apply_modules(*in_dicts)
        assert_equal(result_dict, out_dict)
