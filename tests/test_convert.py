import pcsp
import pytest
from pcsp.module_set import PREV_KEY, MATCH_KEY
from pcsp.convert import combine_dicts

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
                    PREV_KEY:'prev0'
                },
                {('X_test',):'X_test_data', PREV_KEY:'prev1'}
            ],
            # out_dict
            {
                PREV_KEY:['prev0','prev1'], MATCH_KEY:0,
                ('X_train','y_train','RF','X_test'):['RF_fitted','X_test_data'],
                ('X_train','y_train','LR','X_test'):['LR_fitted','X_test_data']
            }
        ),
        (
            # in_dicts
            [
                {
                    PREV_KEY:['prev0','prev1'], MATCH_KEY:0,
                    ('X_train','y_train','RF','X_test'):['RF_fitted','X_test_data'],
                    ('X_train','y_train','LR','X_test'):['LR_fitted','X_test_data']
                },
                {
                    ('y_test',):'y_test_data', ('y_test',):'y_test_data',
                    PREV_KEY:'prev2',MATCH_KEY:0
                }
            ],
            # out_dict
            {
                PREV_KEY:['prev0','prev1','prev2'],
                MATCH_KEY:0,
                ('X_train','y_train','RF','X_test','y_test'):[
                    ['RF_fitted','X_test_data'],'y_test_data'
                ],
                ('X_train','y_train','LR','X_test','y_test'):[
                    ['LR_fitted','X_test_data'],'y_test_data'
                ]
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
                ('X_train','y_train','subsampling_0','RF','X_test','y_test'):[
                    ['RF_fitted_0','X_test_data'],'y_test_data'
                ],
                ('X_train','y_train','subsampling_1','RF','X_test','y_test'):[
                    ['RF_fitted_1','X_test_data'],'y_test_data'
                ],
                ('X_train','y_train','subsampling_0','LR','X_test','y_test'):[
                    ['LR_fitted_0','X_test_data'],'y_test_data'
                ],
                ('X_train','y_train','subsampling_1','LR','X_test','y_test'):[
                    ['LR_fitted_1','X_test_data'],'y_test_data'
                ],
                PREV_KEY:[],
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
                PREV_KEY:[],
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
                    PREV_KEY:[],
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
                ('X_train','y_train','subgroup_0','voxel_extract_0','RF','X_test','y_test'):[
                    ['RF_fitted_00','X_test_data_00'],'y_test_data_00'
                ],
                ('X_train','y_train','subgroup_0','voxel_extract_1','RF','X_test','y_test'):[
                    ['RF_fitted_01','X_test_data_01'],'y_test_data_01'
                ],
                ('X_train','y_train','subgroup_1','voxel_extract_0','RF','X_test','y_test'):[
                    ['RF_fitted_10','X_test_data_10'],'y_test_data_10'
                ],
                ('X_train','y_train','subgroup_1','voxel_extract_1','RF','X_test','y_test'):[
                    ['RF_fitted_11','X_test_data_11'],'y_test_data_11'
                ],
                ('X_train','y_train','subgroup_0','voxel_extract_0','LR','X_test','y_test'):[
                    ['LR_fitted_00','X_test_data_00'],'y_test_data_00'
                ],
                ('X_train','y_train','subgroup_0','voxel_extract_1','LR','X_test','y_test'):[
                    ['LR_fitted_01','X_test_data_01'],'y_test_data_01'
                ],
                ('X_train','y_train','subgroup_1','voxel_extract_0','LR','X_test','y_test'):[
                    ['LR_fitted_10','X_test_data_10'],'y_test_data_10'
                ],
                ('X_train','y_train','subgroup_1','voxel_extract_1','LR','X_test','y_test'):[
                    ['LR_fitted_11','X_test_data_11'],'y_test_data_11'
                ],
                PREV_KEY:[],
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
                    PREV_KEY:[],
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
                ('X_train','y_train','subgroup_0','voxel_extract_0','RF','X_test','y_test','acc'):[
                    [['RF_fitted_00','X_test_data_00'],'y_test_data'],'RF_acc_func'
                ],
                ('X_train','y_train','subgroup_0','voxel_extract_0','RF','X_test','y_test','feat_imp'):[
                    [['RF_fitted_00','X_test_data_00'],'y_test_data'],'RF_feat_imp_func'
                ],
                ('X_train','y_train','subgroup_0','voxel_extract_1','RF','X_test','y_test','acc'):[
                    [['RF_fitted_01','X_test_data_01'],'y_test_data'],'RF_acc_func'
                ],
                ('X_train','y_train','subgroup_0','voxel_extract_1','RF','X_test','y_test','feat_imp'):[
                    [['RF_fitted_01','X_test_data_01'],'y_test_data'],'RF_feat_imp_func'
                ],
                ('X_train','y_train','subgroup_1','voxel_extract_0','RF','X_test','y_test','acc'):[
                    [['RF_fitted_10','X_test_data_10'],'y_test_data'],'RF_acc_func'
                ],
                ('X_train','y_train','subgroup_1','voxel_extract_0','RF','X_test','y_test','feat_imp'):[
                    [['RF_fitted_10','X_test_data_10'],'y_test_data'],'RF_feat_imp_func'
                ],
                ('X_train','y_train','subgroup_1','voxel_extract_1','RF','X_test','y_test','acc'):[
                    [['RF_fitted_11','X_test_data_11'],'y_test_data'],'RF_acc_func'
                ],
                ('X_train','y_train','subgroup_1','voxel_extract_1','RF','X_test','y_test','feat_imp'):[
                    [['RF_fitted_11','X_test_data_11'],'y_test_data'],'RF_feat_imp_func'
                ],
                ('X_train','y_train','subgroup_0','voxel_extract_0','LR','X_test','y_test','acc'):[
                    [['LR_fitted_00','X_test_data_00'],'y_test_data'],'LR_acc_func'
                ],
                ('X_train','y_train','subgroup_0','voxel_extract_0','LR','X_test','y_test','bal_acc'):[
                    [['LR_fitted_00','X_test_data_00'],'y_test_data'],'LR_bal_acc_func'
                ],
                ('X_train','y_train','subgroup_0','voxel_extract_1','LR','X_test','y_test','acc'):[
                    [['LR_fitted_01','X_test_data_01'],'y_test_data'],'LR_acc_func'
                ],
                ('X_train','y_train','subgroup_0','voxel_extract_1','LR','X_test','y_test','bal_acc'):[
                    [['LR_fitted_01','X_test_data_01'],'y_test_data'],'LR_bal_acc_func'
                ],
                ('X_train','y_train','subgroup_1','voxel_extract_0','LR','X_test','y_test','acc'):[
                    [['LR_fitted_10','X_test_data_10'],'y_test_data'],'LR_acc_func'
                ],
                ('X_train','y_train','subgroup_1','voxel_extract_0','LR','X_test','y_test','bal_acc'):[
                    [['LR_fitted_10','X_test_data_10'],'y_test_data'],'LR_bal_acc_func'
                ],
                ('X_train','y_train','subgroup_1','voxel_extract_1','LR','X_test','y_test','acc'):[
                    [['LR_fitted_11','X_test_data_11'],'y_test_data'],'LR_acc_func'
                ],
                ('X_train','y_train','subgroup_1','voxel_extract_1','LR','X_test','y_test','bal_acc'):[
                    [['LR_fitted_11','X_test_data_11'],'y_test_data'],'LR_bal_acc_func'
                ],
                PREV_KEY:[],
                MATCH_KEY:0
            }
        )
    ]
)
class TestCombineTwoDicts:

    # NOTE: these tests rely on consistent iteration of dict keys, which is the
    # case for Python >= 3.7:
    # https://docs.python.org/3/library/stdtypes.html#dictionary-view-objects
    def test_combine_two_dicts(self, in_dicts, out_dict):
        result_dict = combine_dicts(*in_dicts)
        assert result_dict == out_dict
