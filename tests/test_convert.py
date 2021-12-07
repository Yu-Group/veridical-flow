import numpy as np
import pytest
from numpy.testing import assert_equal

from vflow.convert import *
from vflow.subkey import Subkey as sm
import pandas as pd


@pytest.mark.parametrize(
    'in_dicts,out_dict',
    [
        # first or second dict has only one key
        (
                # in_dicts
                [
                    {
                        (sm('X_train', 'init'), sm('y_train', 'init'), sm('RF', 'modeling')): 'RF_fitted',
                        (sm('X_train', 'init'), sm('y_train', 'init'), sm('LR', 'modeling')): 'LR_fitted',
                    },
                    {(sm('X_test', 'init'),): 'X_test_data'}
                ],
                # out_dict
                {
                    (sm('X_train', 'init'), sm('y_train', 'init'), sm('RF', 'modeling'), sm('X_test', 'init')): (
                            'RF_fitted', 'X_test_data'),
                    (sm('X_train', 'init'), sm('y_train', 'init'), sm('LR', 'modeling'), sm('X_test', 'init')): (
                            'LR_fitted', 'X_test_data')
                }
        ),
        (
                # in_dicts
                [
                    {
                        PREV_KEY: ('prev_0', 'prev_1',),
                        (sm('X_train', 'init'), sm('y_train', 'init'), sm('RF', 'modeling'), sm('X_test', 'init')): [
                            'RF_fitted', 'X_test_data'],
                        (sm('X_train', 'init'), sm('y_train', 'init'), sm('LR', 'modeling'), sm('X_test', 'init')): [
                            'LR_fitted', 'X_test_data']
                    },
                    {
                        (sm('y_test', 'init'),): 'y_test_data', (sm('y_test', 'init'),): 'y_test_data',
                        PREV_KEY: ('prev_2',),
                    }
                ],
                # out_dict
                {
                    (sm('X_train', 'init'), sm('y_train', 'init'), sm('RF', 'modeling'), sm('X_test', 'init'),
                     sm('y_test', 'init')): (
                            ['RF_fitted', 'X_test_data'], 'y_test_data'
                    ),
                    (sm('X_train', 'init'), sm('y_train', 'init'), sm('LR', 'modeling'), sm('X_test', 'init'),
                     sm('y_test', 'init')): (
                            ['LR_fitted', 'X_test_data'], 'y_test_data'
                    )
                }
        ),
        (
                # in_dicts
                [
                    {(sm('X_train', 'init'), sm('y_train', 'init'), sm('subsampling_0', 'subsample'),
                      sm('RF', 'modeling')): 'RF_fitted_0',
                     (sm('X_train', 'init'), sm('y_train', 'init'), sm('subsampling_1', 'subsample'),
                      sm('RF', 'modeling')): 'RF_fitted_1',
                     (sm('X_train', 'init'), sm('y_train', 'init'), sm('subsampling_0', 'subsample'),
                      sm('LR', 'modeling')): 'LR_fitted_0',
                     (sm('X_train', 'init'), sm('y_train', 'init'), sm('subsampling_1', 'subsample'),
                      sm('LR', 'modeling')): 'LR_fitted_1'},
                    {(sm('X_test', 'init'),): 'X_test_data'},
                    {(sm('y_test', 'init'),): 'y_test_data'}

                ],
                # out_dict
                {
                    (sm('X_train', 'init'), sm('y_train', 'init'), sm('subsampling_0', 'subsample'),
                     sm('RF', 'modeling'), sm('X_test', 'init'), sm('y_test', 'init')): (
                            'RF_fitted_0', 'X_test_data', 'y_test_data'
                    ),
                    (sm('X_train', 'init'), sm('y_train', 'init'), sm('subsampling_1', 'subsample'),
                     sm('RF', 'modeling'), sm('X_test', 'init'), sm('y_test', 'init')): (
                            'RF_fitted_1', 'X_test_data', 'y_test_data'
                    ),
                    (sm('X_train', 'init'), sm('y_train', 'init'), sm('subsampling_0', 'subsample'),
                     sm('LR', 'modeling'), sm('X_test', 'init'), sm('y_test', 'init')): (
                            'LR_fitted_0', 'X_test_data', 'y_test_data'
                    ),
                    (sm('X_train', 'init'), sm('y_train', 'init'), sm('subsampling_1', 'subsample'),
                     sm('LR', 'modeling'), sm('X_test', 'init'), sm('y_test', 'init')): (
                            'LR_fitted_1', 'X_test_data', 'y_test_data'
                    ),
                }
        ),
        (
                # in_dicts
                [
                    {
                        (sm('X_train', 'init'), sm('y_train', 'init'), sm('subsampling_0', 'origin_0', True),
                         sm('RF', 'modeling')): 'RF_fitted_0',
                        (sm('X_train', 'init'), sm('y_train', 'init'), sm('subsampling_1', 'origin_0', True),
                         sm('RF', 'modeling')): 'RF_fitted_1',
                        (sm('X_train', 'init'), sm('y_train', 'init'), sm('subsampling_0', 'origin_0', True),
                         sm('LR', 'modeling')): 'LR_fitted_0',
                        (sm('X_train', 'init'), sm('y_train', 'init'), sm('subsampling_1', 'origin_0', True),
                         sm('LR', 'modeling')): 'LR_fitted_1',
                    },
                    {
                        (sm('X_train', 'init'), sm('subsampling_0', 'origin_0', True)): 'X_train_data_0',
                        (sm('X_train', 'init'), sm('subsampling_1', 'origin_0', True)): 'X_train_data_1',
                    },
                    {
                        (sm('y_train', 'init'), sm('subsampling_0', 'origin_0', True)): 'y_train_data_0',
                        (sm('y_train', 'init'), sm('subsampling_1', 'origin_0', True)): 'y_train_data_1',
                    }
                ],
                # out_dict
                {
                    (sm('X_train', 'init'), sm('y_train', 'init'), sm('subsampling_0', 'origin_0', True),
                     sm('RF', 'modeling'), sm('X_train', 'init'), sm('y_train', 'init')): (
                            'RF_fitted_0', 'X_train_data_0', 'y_train_data_0'
                    ),
                    (sm('X_train', 'init'), sm('y_train', 'init'), sm('subsampling_1', 'origin_0', True),
                     sm('RF', 'modeling'), sm('X_train', 'init'), sm('y_train', 'init')): (
                            'RF_fitted_1', 'X_train_data_1', 'y_train_data_1'
                    ),
                    (sm('X_train', 'init'), sm('y_train', 'init'), sm('subsampling_0', 'origin_0', True),
                     sm('LR', 'modeling'), sm('X_train', 'init'), sm('y_train', 'init')): (
                            'LR_fitted_0', 'X_train_data_0', 'y_train_data_0'
                    ),
                    (sm('X_train', 'init'), sm('y_train', 'init'), sm('subsampling_1', 'origin_0', True),
                     sm('LR', 'modeling'), sm('X_train', 'init'), sm('y_train', 'init')): (
                            'LR_fitted_1', 'X_train_data_1', 'y_train_data_1'
                    ),
                }
        ),
        (
                # in_dicts
                [
                    {
                        (sm('X_train', 'init'), sm('y_train', 'init'), sm('subgroup_0', 's_origin', True),
                         sm('voxel_extract_0', 'v_origin', True), sm('RF', 'modeling')): 'RF_fitted_00',
                        (sm('X_train', 'init'), sm('y_train', 'init'), sm('subgroup_0', 's_origin', True),
                         sm('voxel_extract_1', 'v_origin', True), sm('RF', 'modeling')): 'RF_fitted_01',
                        (sm('X_train', 'init'), sm('y_train', 'init'), sm('subgroup_1', 's_origin', True),
                         sm('voxel_extract_0', 'v_origin', True), sm('RF', 'modeling')): 'RF_fitted_10',
                        (sm('X_train', 'init'), sm('y_train', 'init'), sm('subgroup_1', 's_origin', True),
                         sm('voxel_extract_1', 'v_origin', True), sm('RF', 'modeling')): 'RF_fitted_11',
                        (sm('X_train', 'init'), sm('y_train', 'init'), sm('subgroup_0', 's_origin', True),
                         sm('voxel_extract_0', 'v_origin', True), sm('LR', 'modeling')): 'LR_fitted_00',
                        (sm('X_train', 'init'), sm('y_train', 'init'), sm('subgroup_0', 's_origin', True),
                         sm('voxel_extract_1', 'v_origin', True), sm('LR', 'modeling')): 'LR_fitted_01',
                        (sm('X_train', 'init'), sm('y_train', 'init'), sm('subgroup_1', 's_origin', True),
                         sm('voxel_extract_0', 'v_origin', True), sm('LR', 'modeling')): 'LR_fitted_10',
                        (sm('X_train', 'init'), sm('y_train', 'init'), sm('subgroup_1', 's_origin', True),
                         sm('voxel_extract_1', 'v_origin', True), sm('LR', 'modeling')): 'LR_fitted_11'
                    },
                    {
                        (sm('X_test', 'init'), sm('subgroup_0', 's_origin', True),
                         sm('voxel_extract_0', 'v_origin', True)): 'X_test_data_00',
                        (sm('X_test', 'init'), sm('subgroup_0', 's_origin', True),
                         sm('voxel_extract_1', 'v_origin', True)): 'X_test_data_01',
                        (sm('X_test', 'init'), sm('subgroup_1', 's_origin', True),
                         sm('voxel_extract_0', 'v_origin', True)): 'X_test_data_10',
                        (sm('X_test', 'init'), sm('subgroup_1', 's_origin', True),
                         sm('voxel_extract_1', 'v_origin', True)): 'X_test_data_11'
                    }
                ],
                # out_dict
                {
                    (sm('X_train', 'init'), sm('y_train', 'init'), sm('subgroup_0', 's_origin', True),
                     sm('voxel_extract_0', 'v_origin', True), sm('RF', 'modeling'), sm('X_test', 'init')): (
                            'RF_fitted_00', 'X_test_data_00'
                    ),
                    (sm('X_train', 'init'), sm('y_train', 'init'), sm('subgroup_0', 's_origin', True),
                     sm('voxel_extract_1', 'v_origin', True), sm('RF', 'modeling'), sm('X_test', 'init')): (
                            'RF_fitted_01', 'X_test_data_01'
                    ),
                    (sm('X_train', 'init'), sm('y_train', 'init'), sm('subgroup_1', 's_origin', True),
                     sm('voxel_extract_0', 'v_origin', True), sm('RF', 'modeling'), sm('X_test', 'init')): (
                            'RF_fitted_10', 'X_test_data_10'
                    ),
                    (sm('X_train', 'init'), sm('y_train', 'init'), sm('subgroup_1', 's_origin', True),
                     sm('voxel_extract_1', 'v_origin', True), sm('RF', 'modeling'), sm('X_test', 'init')): (
                            'RF_fitted_11', 'X_test_data_11'
                    ),
                    (sm('X_train', 'init'), sm('y_train', 'init'), sm('subgroup_0', 's_origin', True),
                     sm('voxel_extract_0', 'v_origin', True), sm('LR', 'modeling'), sm('X_test', 'init')): (
                            'LR_fitted_00', 'X_test_data_00'
                    ),
                    (sm('X_train', 'init'), sm('y_train', 'init'), sm('subgroup_0', 's_origin', True),
                     sm('voxel_extract_1', 'v_origin', True), sm('LR', 'modeling'), sm('X_test', 'init')): (
                            'LR_fitted_01', 'X_test_data_01'
                    ),
                    (sm('X_train', 'init'), sm('y_train', 'init'), sm('subgroup_1', 's_origin', True),
                     sm('voxel_extract_0', 'v_origin', True), sm('LR', 'modeling'), sm('X_test', 'init')): (
                            'LR_fitted_10', 'X_test_data_10'
                    ),
                    (sm('X_train', 'init'), sm('y_train', 'init'), sm('subgroup_1', 's_origin', True),
                     sm('voxel_extract_1', 'v_origin', True), sm('LR', 'modeling'), sm('X_test', 'init')): (
                            'LR_fitted_11', 'X_test_data_11'
                    ),
                }
        ),
        (
                # in_dicts
                [
                    {
                        (sm('X_train', 'init'), sm('y_train', 'init'), sm('subgroup_0', 's_origin', True),
                         sm('voxel_extract_0', 'v_origin', True), sm('RF', 'modeling')): 'RF_fitted_00',
                        (sm('X_train', 'init'), sm('y_train', 'init'), sm('subgroup_0', 's_origin', True),
                         sm('voxel_extract_1', 'v_origin', True), sm('RF', 'modeling')): 'RF_fitted_01',
                        (sm('X_train', 'init'), sm('y_train', 'init'), sm('subgroup_1', 's_origin', True),
                         sm('voxel_extract_0', 'v_origin', True), sm('RF', 'modeling')): 'RF_fitted_10',
                        (sm('X_train', 'init'), sm('y_train', 'init'), sm('subgroup_1', 's_origin', True),
                         sm('voxel_extract_1', 'v_origin', True), sm('RF', 'modeling')): 'RF_fitted_11',
                        (sm('X_train', 'init'), sm('y_train', 'init'), sm('subgroup_0', 's_origin', True),
                         sm('voxel_extract_0', 'v_origin', True), sm('LR', 'modeling')): 'LR_fitted_00',
                        (sm('X_train', 'init'), sm('y_train', 'init'), sm('subgroup_0', 's_origin', True),
                         sm('voxel_extract_1', 'v_origin', True), sm('LR', 'modeling')): 'LR_fitted_01',
                        (sm('X_train', 'init'), sm('y_train', 'init'), sm('subgroup_1', 's_origin', True),
                         sm('voxel_extract_0', 'v_origin', True), sm('LR', 'modeling')): 'LR_fitted_10',
                        (sm('X_train', 'init'), sm('y_train', 'init'), sm('subgroup_1', 's_origin', True),
                         sm('voxel_extract_1', 'v_origin', True), sm('LR', 'modeling')): 'LR_fitted_11'
                    },
                    {
                        (sm('X_test', 'init'), sm('subgroup_0', 's_origin', True),
                         sm('voxel_extract_0', 'v_origin', True)): 'X_test_data_00',
                        (sm('X_test', 'init'), sm('subgroup_0', 's_origin', True),
                         sm('voxel_extract_1', 'v_origin', True)): 'X_test_data_01',
                        (sm('X_test', 'init'), sm('subgroup_1', 's_origin', True),
                         sm('voxel_extract_0', 'v_origin', True)): 'X_test_data_10',
                        (sm('X_test', 'init'), sm('subgroup_1', 's_origin', True),
                         sm('voxel_extract_1', 'v_origin', True)): 'X_test_data_11'
                    }
                ],
                # out_dict
                {
                    (sm('X_train', 'init'), sm('y_train', 'init'), sm('subgroup_0', 's_origin', True),
                     sm('voxel_extract_0', 'v_origin', True), sm('RF', 'modeling'), sm('X_test', 'init')): (
                            'RF_fitted_00', 'X_test_data_00'
                    ),
                    (sm('X_train', 'init'), sm('y_train', 'init'), sm('subgroup_0', 's_origin', True),
                     sm('voxel_extract_1', 'v_origin', True), sm('RF', 'modeling'), sm('X_test', 'init')): (
                            'RF_fitted_01', 'X_test_data_01'
                    ),
                    (sm('X_train', 'init'), sm('y_train', 'init'), sm('subgroup_1', 's_origin', True),
                     sm('voxel_extract_0', 'v_origin', True), sm('RF', 'modeling'), sm('X_test', 'init')): (
                            'RF_fitted_10', 'X_test_data_10'
                    ),
                    (sm('X_train', 'init'), sm('y_train', 'init'), sm('subgroup_1', 's_origin', True),
                     sm('voxel_extract_1', 'v_origin', True), sm('RF', 'modeling'), sm('X_test', 'init')): (
                            'RF_fitted_11', 'X_test_data_11'
                    ),
                    (sm('X_train', 'init'), sm('y_train', 'init'), sm('subgroup_0', 's_origin', True),
                     sm('voxel_extract_0', 'v_origin', True), sm('LR', 'modeling'), sm('X_test', 'init')): (
                            'LR_fitted_00', 'X_test_data_00'
                    ),
                    (sm('X_train', 'init'), sm('y_train', 'init'), sm('subgroup_0', 's_origin', True),
                     sm('voxel_extract_1', 'v_origin', True), sm('LR', 'modeling'), sm('X_test', 'init')): (
                            'LR_fitted_01', 'X_test_data_01'
                    ),
                    (sm('X_train', 'init'), sm('y_train', 'init'), sm('subgroup_1', 's_origin', True),
                     sm('voxel_extract_0', 'v_origin', True), sm('LR', 'modeling'), sm('X_test', 'init')): (
                            'LR_fitted_10', 'X_test_data_10'
                    ),
                    (sm('X_train', 'init'), sm('y_train', 'init'), sm('subgroup_1', 's_origin', True),
                     sm('voxel_extract_1', 'v_origin', True), sm('LR', 'modeling'), sm('X_test', 'init')): (
                            'LR_fitted_11', 'X_test_data_11'
                    ),
                }
        ),
        (
                # in_dicts
                [
                    {
                        (sm('X_train', 'init'), sm('y_train', 'init'), sm('subgroup_0', 's_origin', True),
                         sm('voxel_extract_0', 'v_origin', True), sm('RF', 'modeling'), sm('X_test', 'init')): [
                            'RF_fitted_00', 'X_test_data_00'
                        ],
                        (sm('X_train', 'init'), sm('y_train', 'init'), sm('subgroup_0', 's_origin', True),
                         sm('voxel_extract_1', 'v_origin', True), sm('RF', 'modeling'), sm('X_test', 'init')): [
                            'RF_fitted_01', 'X_test_data_01'
                        ],
                        (sm('X_train', 'init'), sm('y_train', 'init'), sm('subgroup_1', 's_origin', True),
                         sm('voxel_extract_0', 'v_origin', True), sm('RF', 'modeling'), sm('X_test', 'init')): [
                            'RF_fitted_10', 'X_test_data_10'
                        ],
                        (sm('X_train', 'init'), sm('y_train', 'init'), sm('subgroup_1', 's_origin', True),
                         sm('voxel_extract_1', 'v_origin', True), sm('RF', 'modeling'), sm('X_test', 'init')): [
                            'RF_fitted_11', 'X_test_data_11'
                        ],
                        (sm('X_train', 'init'), sm('y_train', 'init'), sm('subgroup_0', 's_origin', True),
                         sm('voxel_extract_0', 'v_origin', True), sm('LR', 'modeling'), sm('X_test', 'init')): [
                            'LR_fitted_00', 'X_test_data_00'
                        ],
                        (sm('X_train', 'init'), sm('y_train', 'init'), sm('subgroup_0', 's_origin', True),
                         sm('voxel_extract_1', 'v_origin', True), sm('LR', 'modeling'), sm('X_test', 'init')): [
                            'LR_fitted_01', 'X_test_data_01'
                        ],
                        (sm('X_train', 'init'), sm('y_train', 'init'), sm('subgroup_1', 's_origin', True),
                         sm('voxel_extract_0', 'v_origin', True), sm('LR', 'modeling'), sm('X_test', 'init')): [
                            'LR_fitted_10', 'X_test_data_10'
                        ],
                        (sm('X_train', 'init'), sm('y_train', 'init'), sm('subgroup_1', 's_origin', True),
                         sm('voxel_extract_1', 'v_origin', True), sm('LR', 'modeling'), sm('X_test', 'init')): [
                            'LR_fitted_11', 'X_test_data_11'
                        ],
                    },
                    {
                        (sm('y_test', 'init'), sm('subgroup_0', 's_origin', True),
                         sm('voxel_extract_0', 'v_origin', True)): 'y_test_data_00',
                        (sm('y_test', 'init'), sm('subgroup_0', 's_origin', True),
                         sm('voxel_extract_1', 'v_origin', True)): 'y_test_data_01',
                        (sm('y_test', 'init'), sm('subgroup_1', 's_origin', True),
                         sm('voxel_extract_0', 'v_origin', True)): 'y_test_data_10',
                        (sm('y_test', 'init'), sm('subgroup_1', 's_origin', True),
                         sm('voxel_extract_1', 'v_origin', True)): 'y_test_data_11'
                    }
                ],
                # out_dict
                {
                    (sm('X_train', 'init'), sm('y_train', 'init'), sm('subgroup_0', 's_origin', True),
                     sm('voxel_extract_0', 'v_origin', True), sm('RF', 'modeling'), sm('X_test', 'init'),
                     sm('y_test', 'init')): (
                            ['RF_fitted_00', 'X_test_data_00'], 'y_test_data_00'
                    ),
                    (sm('X_train', 'init'), sm('y_train', 'init'), sm('subgroup_0', 's_origin', True),
                     sm('voxel_extract_1', 'v_origin', True), sm('RF', 'modeling'), sm('X_test', 'init'),
                     sm('y_test', 'init')): (
                            ['RF_fitted_01', 'X_test_data_01'], 'y_test_data_01'
                    ),
                    (sm('X_train', 'init'), sm('y_train', 'init'), sm('subgroup_1', 's_origin', True),
                     sm('voxel_extract_0', 'v_origin', True), sm('RF', 'modeling'), sm('X_test', 'init'),
                     sm('y_test', 'init')): (
                            ['RF_fitted_10', 'X_test_data_10'], 'y_test_data_10'
                    ),
                    (sm('X_train', 'init'), sm('y_train', 'init'), sm('subgroup_1', 's_origin', True),
                     sm('voxel_extract_1', 'v_origin', True), sm('RF', 'modeling'), sm('X_test', 'init'),
                     sm('y_test', 'init')): (
                            ['RF_fitted_11', 'X_test_data_11'], 'y_test_data_11'
                    ),
                    (sm('X_train', 'init'), sm('y_train', 'init'), sm('subgroup_0', 's_origin', True),
                     sm('voxel_extract_0', 'v_origin', True), sm('LR', 'modeling'), sm('X_test', 'init'),
                     sm('y_test', 'init')): (
                            ['LR_fitted_00', 'X_test_data_00'], 'y_test_data_00'
                    ),
                    (sm('X_train', 'init'), sm('y_train', 'init'), sm('subgroup_0', 's_origin', True),
                     sm('voxel_extract_1', 'v_origin', True), sm('LR', 'modeling'), sm('X_test', 'init'),
                     sm('y_test', 'init')): (
                            ['LR_fitted_01', 'X_test_data_01'], 'y_test_data_01'
                    ),
                    (sm('X_train', 'init'), sm('y_train', 'init'), sm('subgroup_1', 's_origin', True),
                     sm('voxel_extract_0', 'v_origin', True), sm('LR', 'modeling'), sm('X_test', 'init'),
                     sm('y_test', 'init')): (
                            ['LR_fitted_10', 'X_test_data_10'], 'y_test_data_10'
                    ),
                    (sm('X_train', 'init'), sm('y_train', 'init'), sm('subgroup_1', 's_origin', True),
                     sm('voxel_extract_1', 'v_origin', True), sm('LR', 'modeling'), sm('X_test', 'init'),
                     sm('y_test', 'init')): (
                            ['LR_fitted_11', 'X_test_data_11'], 'y_test_data_11'
                    ),
                }
        ),
        (
                # in_dicts
                [
                    {
                        (sm('X_train', 'init'), sm('y_train', 'init'), sm('subgroup_0', 's_origin', True),
                         sm('voxel_extract_0', 'v_origin', True), sm('RF', 'm_origin', True), sm('X_test', 'init'),
                         sm('y_test', 'init')): [
                            ['RF_fitted_00', 'X_test_data_00'], 'y_test_data'
                        ],
                        (sm('X_train', 'init'), sm('y_train', 'init'), sm('subgroup_0', 's_origin', True),
                         sm('voxel_extract_1', 'v_origin', True), sm('RF', 'm_origin', True), sm('X_test', 'init'),
                         sm('y_test', 'init')): [
                            ['RF_fitted_01', 'X_test_data_01'], 'y_test_data'
                        ],
                        (sm('X_train', 'init'), sm('y_train', 'init'), sm('subgroup_1', 's_origin', True),
                         sm('voxel_extract_0', 'v_origin', True), sm('RF', 'm_origin', True), sm('X_test', 'init'),
                         sm('y_test', 'init')): [
                            ['RF_fitted_10', 'X_test_data_10'], 'y_test_data'
                        ],
                        (sm('X_train', 'init'), sm('y_train', 'init'), sm('subgroup_1', 's_origin', True),
                         sm('voxel_extract_1', 'v_origin', True), sm('RF', 'm_origin', True), sm('X_test', 'init'),
                         sm('y_test', 'init')): [
                            ['RF_fitted_11', 'X_test_data_11'], 'y_test_data'
                        ],
                        (sm('X_train', 'init'), sm('y_train', 'init'), sm('subgroup_0', 's_origin', True),
                         sm('voxel_extract_0', 'v_origin', True), sm('LR', 'm_origin', True), sm('X_test', 'init'),
                         sm('y_test', 'init')): [
                            ['LR_fitted_00', 'X_test_data_00'], 'y_test_data'
                        ],
                        (sm('X_train', 'init'), sm('y_train', 'init'), sm('subgroup_0', 's_origin', True),
                         sm('voxel_extract_1', 'v_origin', True), sm('LR', 'm_origin', True), sm('X_test', 'init'),
                         sm('y_test', 'init')): [
                            ['LR_fitted_01', 'X_test_data_01'], 'y_test_data'
                        ],
                        (sm('X_train', 'init'), sm('y_train', 'init'), sm('subgroup_1', 's_origin', True),
                         sm('voxel_extract_0', 'v_origin', True), sm('LR', 'm_origin', True), sm('X_test', 'init'),
                         sm('y_test', 'init')): [
                            ['LR_fitted_10', 'X_test_data_10'], 'y_test_data'
                        ],
                        (sm('X_train', 'init'), sm('y_train', 'init'), sm('subgroup_1', 's_origin', True),
                         sm('voxel_extract_1', 'v_origin', True), sm('LR', 'm_origin', True), sm('X_test', 'init'),
                         sm('y_test', 'init')): [
                            ['LR_fitted_11', 'X_test_data_11'], 'y_test_data'
                        ],
                    },
                    {
                        (sm('LR', 'm_origin', True), sm('acc', 'metrics')): 'LR_acc_func',
                        (sm('LR', 'm_origin', True), sm('bal_acc', 'metrics')): 'LR_bal_acc_func',
                        (sm('RF', 'm_origin', True), sm('acc', 'metrics')): 'RF_acc_func',
                        (sm('RF', 'm_origin', True), sm('feat_imp', 'metrics')): 'RF_feat_imp_func'
                    }
                ],
                # out_dict
                {
                    (sm('X_train', 'init'), sm('y_train', 'init'), sm('subgroup_0', 's_origin', True),
                     sm('voxel_extract_0', 'v_origin', True), sm('RF', 'm_origin', True), sm('X_test', 'init'),
                     sm('y_test', 'init'), sm('acc', 'metrics')): (
                            [['RF_fitted_00', 'X_test_data_00'], 'y_test_data'], 'RF_acc_func'
                    ),
                    (sm('X_train', 'init'), sm('y_train', 'init'), sm('subgroup_0', 's_origin', True),
                     sm('voxel_extract_0', 'v_origin', True), sm('RF', 'm_origin', True), sm('X_test', 'init'),
                     sm('y_test', 'init'), sm('feat_imp', 'metrics')): (
                            [['RF_fitted_00', 'X_test_data_00'], 'y_test_data'], 'RF_feat_imp_func'
                    ),
                    (sm('X_train', 'init'), sm('y_train', 'init'), sm('subgroup_0', 's_origin', True),
                     sm('voxel_extract_1', 'v_origin', True), sm('RF', 'm_origin', True), sm('X_test', 'init'),
                     sm('y_test', 'init'), sm('acc', 'metrics')): (
                            [['RF_fitted_01', 'X_test_data_01'], 'y_test_data'], 'RF_acc_func'
                    ),
                    (sm('X_train', 'init'), sm('y_train', 'init'), sm('subgroup_0', 's_origin', True),
                     sm('voxel_extract_1', 'v_origin', True), sm('RF', 'm_origin', True), sm('X_test', 'init'),
                     sm('y_test', 'init'), sm('feat_imp', 'metrics')): (
                            [['RF_fitted_01', 'X_test_data_01'], 'y_test_data'], 'RF_feat_imp_func'
                    ),
                    (sm('X_train', 'init'), sm('y_train', 'init'), sm('subgroup_1', 's_origin', True),
                     sm('voxel_extract_0', 'v_origin', True), sm('RF', 'm_origin', True), sm('X_test', 'init'),
                     sm('y_test', 'init'), sm('acc', 'metrics')): (
                            [['RF_fitted_10', 'X_test_data_10'], 'y_test_data'], 'RF_acc_func'
                    ),
                    (sm('X_train', 'init'), sm('y_train', 'init'), sm('subgroup_1', 's_origin', True),
                     sm('voxel_extract_0', 'v_origin', True), sm('RF', 'm_origin', True), sm('X_test', 'init'),
                     sm('y_test', 'init'), sm('feat_imp', 'metrics')): (
                            [['RF_fitted_10', 'X_test_data_10'], 'y_test_data'], 'RF_feat_imp_func'
                    ),
                    (sm('X_train', 'init'), sm('y_train', 'init'), sm('subgroup_1', 's_origin', True),
                     sm('voxel_extract_1', 'v_origin', True), sm('RF', 'm_origin', True), sm('X_test', 'init'),
                     sm('y_test', 'init'), sm('acc', 'metrics')): (
                            [['RF_fitted_11', 'X_test_data_11'], 'y_test_data'], 'RF_acc_func'
                    ),
                    (sm('X_train', 'init'), sm('y_train', 'init'), sm('subgroup_1', 's_origin', True),
                     sm('voxel_extract_1', 'v_origin', True), sm('RF', 'm_origin', True), sm('X_test', 'init'),
                     sm('y_test', 'init'), sm('feat_imp', 'metrics')): (
                            [['RF_fitted_11', 'X_test_data_11'], 'y_test_data'], 'RF_feat_imp_func'
                    ),
                    (sm('X_train', 'init'), sm('y_train', 'init'), sm('subgroup_0', 's_origin', True),
                     sm('voxel_extract_0', 'v_origin', True), sm('LR', 'm_origin', True), sm('X_test', 'init'),
                     sm('y_test', 'init'), sm('acc', 'metrics')): (
                            [['LR_fitted_00', 'X_test_data_00'], 'y_test_data'], 'LR_acc_func'
                    ),
                    (sm('X_train', 'init'), sm('y_train', 'init'), sm('subgroup_0', 's_origin', True),
                     sm('voxel_extract_0', 'v_origin', True), sm('LR', 'm_origin', True), sm('X_test', 'init'),
                     sm('y_test', 'init'), sm('bal_acc', 'metrics')): (
                            [['LR_fitted_00', 'X_test_data_00'], 'y_test_data'], 'LR_bal_acc_func'
                    ),
                    (sm('X_train', 'init'), sm('y_train', 'init'), sm('subgroup_0', 's_origin', True),
                     sm('voxel_extract_1', 'v_origin', True), sm('LR', 'm_origin', True), sm('X_test', 'init'),
                     sm('y_test', 'init'), sm('acc', 'metrics')): (
                            [['LR_fitted_01', 'X_test_data_01'], 'y_test_data'], 'LR_acc_func'
                    ),
                    (sm('X_train', 'init'), sm('y_train', 'init'), sm('subgroup_0', 's_origin', True),
                     sm('voxel_extract_1', 'v_origin', True), sm('LR', 'm_origin', True), sm('X_test', 'init'),
                     sm('y_test', 'init'), sm('bal_acc', 'metrics')): (
                            [['LR_fitted_01', 'X_test_data_01'], 'y_test_data'], 'LR_bal_acc_func'
                    ),
                    (sm('X_train', 'init'), sm('y_train', 'init'), sm('subgroup_1', 's_origin', True),
                     sm('voxel_extract_0', 'v_origin', True), sm('LR', 'm_origin', True), sm('X_test', 'init'),
                     sm('y_test', 'init'), sm('acc', 'metrics')): (
                            [['LR_fitted_10', 'X_test_data_10'], 'y_test_data'], 'LR_acc_func'
                    ),
                    (sm('X_train', 'init'), sm('y_train', 'init'), sm('subgroup_1', 's_origin', True),
                     sm('voxel_extract_0', 'v_origin', True), sm('LR', 'm_origin', True), sm('X_test', 'init'),
                     sm('y_test', 'init'), sm('bal_acc', 'metrics')): (
                            [['LR_fitted_10', 'X_test_data_10'], 'y_test_data'], 'LR_bal_acc_func'
                    ),
                    (sm('X_train', 'init'), sm('y_train', 'init'), sm('subgroup_1', 's_origin', True),
                     sm('voxel_extract_1', 'v_origin', True), sm('LR', 'm_origin', True), sm('X_test', 'init'),
                     sm('y_test', 'init'), sm('acc', 'metrics')): (
                            [['LR_fitted_11', 'X_test_data_11'], 'y_test_data'], 'LR_acc_func'
                    ),
                    (sm('X_train', 'init'), sm('y_train', 'init'), sm('subgroup_1', 's_origin', True),
                     sm('voxel_extract_1', 'v_origin', True), sm('LR', 'm_origin', True), sm('X_test', 'init'),
                     sm('y_test', 'init'), sm('bal_acc', 'metrics')): (
                            [['LR_fitted_11', 'X_test_data_11'], 'y_test_data'], 'LR_bal_acc_func'
                    ),
                }
        ),
        (
                # in_dicts
                [
                    {
                        (sm('X_train', 'init'), sm('y_train', 'init'), sm('voxel_extract_0', 'v_origin', True),
                         sm('subgroup_0', 's_origin', True), sm('RF', 'modeling')): 'RF_00',
                        (sm('X_train', 'init'), sm('y_train', 'init'), sm('voxel_extract_0', 'v_origin', True),
                         sm('subgroup_1', 's_origin', True), sm('RF', 'modeling')): 'RF_01',
                        (sm('X_train', 'init'), sm('y_train', 'init'), sm('voxel_extract_1', 'v_origin', True),
                         sm('subgroup_0', 's_origin', True), sm('RF', 'modeling')): 'RF_10',
                        (sm('X_train', 'init'), sm('y_train', 'init'), sm('voxel_extract_1', 'v_origin', True),
                         sm('subgroup_1', 's_origin', True), sm('RF', 'modeling')): 'RF_11',
                    },
                    {
                        (sm('X_test', 'init'), sm('subgroup_0', 's_origin', True)): 'X_test_data_0',
                        (sm('X_test', 'init'), sm('subgroup_1', 's_origin', True)): 'X_test_data_1'
                    },
                    {
                        (sm('y_test', 'init'), sm('voxel_extract_0', 'v_origin', True),
                         sm('subgroup_0', 's_origin', True)): 'y_test_data_00',
                        (sm('y_test', 'init'), sm('voxel_extract_0', 'v_origin', True),
                         sm('subgroup_1', 's_origin', True)): 'y_test_data_01',
                        (sm('y_test', 'init'), sm('voxel_extract_1', 'v_origin', True),
                         sm('subgroup_0', 's_origin', True)): 'y_test_data_10',
                        (sm('y_test', 'init'), sm('voxel_extract_1', 'v_origin', True),
                         sm('subgroup_1', 's_origin', True)): 'y_test_data_11',
                    }
                ],
                # out_dict
                {
                    (sm('X_train', 'init'), sm('y_train', 'init'), sm('voxel_extract_0', 'v_origin', True),
                     sm('subgroup_0', 's_origin', True),
                     sm('RF', 'modeling'), sm('X_test', 'init'), sm('y_test', 'init')): (
                            'RF_00', 'X_test_data_0', 'y_test_data_00'
                    ),
                    (sm('X_train', 'init'), sm('y_train', 'init'), sm('voxel_extract_0', 'v_origin', True),
                     sm('subgroup_1', 's_origin', True),
                     sm('RF', 'modeling'), sm('X_test', 'init'), sm('y_test', 'init')): (
                            'RF_01', 'X_test_data_1', 'y_test_data_01'
                    ),
                    (sm('X_train', 'init'), sm('y_train', 'init'), sm('voxel_extract_1', 'v_origin', True),
                     sm('subgroup_0', 's_origin', True),
                     sm('RF', 'modeling'), sm('X_test', 'init'), sm('y_test', 'init')): (
                            'RF_10', 'X_test_data_0', 'y_test_data_10'
                    ),
                    (sm('X_train', 'init'), sm('y_train', 'init'), sm('voxel_extract_1', 'v_origin', True),
                     sm('subgroup_1', 's_origin', True),
                     sm('RF', 'modeling'), sm('X_test', 'init'), sm('y_test', 'init')): (
                            'RF_11', 'X_test_data_1', 'y_test_data_11'
                    )
                },
        ),
        (
                # in_dicts
                [
                    {
                        (sm('X_train', 'init'), sm('y_train', 'init'), sm('voxel_extract_0', 'v_origin', True),
                         sm('subgroup_0', 's_origin', True), sm('RF', 'modeling')): 'RF_00',
                        (sm('X_train', 'init'), sm('y_train', 'init'), sm('voxel_extract_0', 'v_origin', True),
                         sm('subgroup_1', 's_origin', True), sm('RF', 'modeling')): 'RF_01',
                        (sm('X_train', 'init'), sm('y_train', 'init'), sm('voxel_extract_1', 'v_origin', True),
                         sm('subgroup_0', 's_origin', True), sm('RF', 'modeling')): 'RF_10',
                        (sm('X_train', 'init'), sm('y_train', 'init'), sm('voxel_extract_1', 'v_origin', True),
                         sm('subgroup_1', 's_origin', True), sm('RF', 'modeling')): 'RF_11',
                    },
                    {
                        (sm('y_test', 'init'), sm('voxel_extract_0', 'v_origin', True),
                         sm('subgroup_0', 's_origin', True)): 'y_test_data_00',
                        (sm('y_test', 'init'), sm('voxel_extract_0', 'v_origin', True),
                         sm('subgroup_1', 's_origin', True)): 'y_test_data_01',
                        (sm('y_test', 'init'), sm('voxel_extract_1', 'v_origin', True),
                         sm('subgroup_0', 's_origin', True)): 'y_test_data_10',
                        (sm('y_test', 'init'), sm('voxel_extract_1', 'v_origin', True),
                         sm('subgroup_1', 's_origin', True)): 'y_test_data_11',
                    },
                    {
                        (sm('X_test', 'init'), sm('subgroup_0', 's_origin', True)): 'X_test_data_0',
                        (sm('X_test', 'init'), sm('subgroup_1', 's_origin', True)): 'X_test_data_1'
                    },
                ],
                # out_dict
                {
                    (sm('X_train', 'init'), sm('y_train', 'init'), sm('voxel_extract_0', 'v_origin', True),
                     sm('subgroup_0', 's_origin', True), sm('RF', 'modeling'), sm('y_test', 'init'),
                     sm('X_test', 'init')): (
                            'RF_00', 'y_test_data_00', 'X_test_data_0'
                    ),
                    (sm('X_train', 'init'), sm('y_train', 'init'), sm('voxel_extract_0', 'v_origin', True),
                     sm('subgroup_1', 's_origin', True), sm('RF', 'modeling'), sm('y_test', 'init'),
                     sm('X_test', 'init')): (
                            'RF_01', 'y_test_data_01', 'X_test_data_1'
                    ),
                    (sm('X_train', 'init'), sm('y_train', 'init'), sm('voxel_extract_1', 'v_origin', True),
                     sm('subgroup_0', 's_origin', True), sm('RF', 'modeling'), sm('y_test', 'init'),
                     sm('X_test', 'init')): (
                            'RF_10', 'y_test_data_10', 'X_test_data_0'
                    ),
                    (sm('X_train', 'init'), sm('y_train', 'init'), sm('voxel_extract_1', 'v_origin', True),
                     sm('subgroup_1', 's_origin', True), sm('RF', 'modeling'), sm('y_test', 'init'),
                     sm('X_test', 'init')): (
                            'RF_11', 'y_test_data_11', 'X_test_data_1'
                    )
                },
        ),
        (
                # in_dicts
                [
                    {
                        (sm('X_test', 'init'), sm('subgroup_0', 's_origin', True)): 'X_test_data_0',
                        (sm('X_test', 'init'), sm('subgroup_1', 's_origin', True)): 'X_test_data_1'
                    },
                    {
                        (sm('X_train', 'init'), sm('y_train', 'init'), sm('voxel_extract_0', 'v_origin', True),
                         sm('subgroup_0', 's_origin', True), sm('RF', 'modeling')): 'RF_00',
                        (sm('X_train', 'init'), sm('y_train', 'init'), sm('voxel_extract_0', 'v_origin', True),
                         sm('subgroup_1', 's_origin', True), sm('RF', 'modeling')): 'RF_01',
                        (sm('X_train', 'init'), sm('y_train', 'init'), sm('voxel_extract_1', 'v_origin', True),
                         sm('subgroup_0', 's_origin', True), sm('RF', 'modeling')): 'RF_10',
                        (sm('X_train', 'init'), sm('y_train', 'init'), sm('voxel_extract_1', 'v_origin', True),
                         sm('subgroup_1', 's_origin', True), sm('RF', 'modeling')): 'RF_11',
                    },
                    {
                        (sm('y_test', 'init'), sm('voxel_extract_0', 'v_origin', True),
                         sm('subgroup_0', 's_origin', True)): 'y_test_data_00',
                        (sm('y_test', 'init'), sm('voxel_extract_0', 'v_origin', True),
                         sm('subgroup_1', 's_origin', True)): 'y_test_data_01',
                        (sm('y_test', 'init'), sm('voxel_extract_1', 'v_origin', True),
                         sm('subgroup_0', 's_origin', True)): 'y_test_data_10',
                        (sm('y_test', 'init'), sm('voxel_extract_1', 'v_origin', True),
                         sm('subgroup_1', 's_origin', True)): 'y_test_data_11',
                    }
                ],
                # out_dict
                {
                    (sm('X_test', 'init'), sm('subgroup_0', 's_origin', True), sm('X_train', 'init'),
                     sm('y_train', 'init'),
                     sm('voxel_extract_0', 'v_origin', True), sm('RF', 'modeling'), sm('y_test', 'init')): (
                            'X_test_data_0', 'RF_00', 'y_test_data_00'
                    ),
                    (sm('X_test', 'init'), sm('subgroup_1', 's_origin', True), sm('X_train', 'init'),
                     sm('y_train', 'init'),
                     sm('voxel_extract_0', 'v_origin', True), sm('RF', 'modeling'), sm('y_test', 'init')): (
                            'X_test_data_1', 'RF_01', 'y_test_data_01'
                    ),
                    (sm('X_test', 'init'), sm('subgroup_0', 's_origin', True), sm('X_train', 'init'),
                     sm('y_train', 'init'),
                     sm('voxel_extract_1', 'v_origin', True), sm('RF', 'modeling'), sm('y_test', 'init')): (
                            'X_test_data_0', 'RF_10', 'y_test_data_10'
                    ),
                    (sm('X_test', 'init'), sm('subgroup_1', 's_origin', True), sm('X_train', 'init'),
                     sm('y_train', 'init'),
                     sm('voxel_extract_1', 'v_origin', True), sm('RF', 'modeling'), sm('y_test', 'init')): (
                            'X_test_data_1', 'RF_11', 'y_test_data_11'
                    )
                },
        ),
        (
                # in_dicts
                [
                    {
                        (sm('X_test', 'init'), sm('subgroup_0', 's_origin', True)): 'X_test_data_0',
                        (sm('X_test', 'init'), sm('subgroup_1', 's_origin', True)): 'X_test_data_1'
                    },
                    {
                        (sm('y_test', 'init'), sm('voxel_extract_0', 'v_origin', True),
                         sm('subgroup_0', 's_origin', True)): 'y_test_data_00',
                        (sm('y_test', 'init'), sm('voxel_extract_0', 'v_origin', True),
                         sm('subgroup_1', 's_origin', True)): 'y_test_data_01',
                        (sm('y_test', 'init'), sm('voxel_extract_1', 'v_origin', True),
                         sm('subgroup_0', 's_origin', True)): 'y_test_data_10',
                        (sm('y_test', 'init'), sm('voxel_extract_1', 'v_origin', True),
                         sm('subgroup_1', 's_origin', True)): 'y_test_data_11',
                    },
                    {
                        (sm('X_train', 'init'), sm('y_train', 'init'), sm('voxel_extract_0', 'v_origin', True),
                         sm('subgroup_0', 's_origin', True), sm('RF', 'modeling')): 'RF_00',
                        (sm('X_train', 'init'), sm('y_train', 'init'), sm('voxel_extract_0', 'v_origin', True),
                         sm('subgroup_1', 's_origin', True), sm('RF', 'modeling')): 'RF_01',
                        (sm('X_train', 'init'), sm('y_train', 'init'), sm('voxel_extract_1', 'v_origin', True),
                         sm('subgroup_0', 's_origin', True), sm('RF', 'modeling')): 'RF_10',
                        (sm('X_train', 'init'), sm('y_train', 'init'), sm('voxel_extract_1', 'v_origin', True),
                         sm('subgroup_1', 's_origin', True), sm('RF', 'modeling')): 'RF_11',
                    },
                ],
                # out_dict
                {
                    (sm('X_test', 'init'), sm('subgroup_0', 's_origin', True), sm('y_test', 'init'),
                     sm('voxel_extract_0', 'v_origin', True), sm('X_train', 'init'), sm('y_train', 'init'),
                     sm('RF', 'modeling')): (
                            'X_test_data_0', 'y_test_data_00', 'RF_00'
                    ),
                    (sm('X_test', 'init'), sm('subgroup_1', 's_origin', True), sm('y_test', 'init'),
                     sm('voxel_extract_0', 'v_origin', True), sm('X_train', 'init'), sm('y_train', 'init'),
                     sm('RF', 'modeling')): (
                            'X_test_data_1', 'y_test_data_01', 'RF_01'
                    ),
                    (sm('X_test', 'init'), sm('subgroup_0', 's_origin', True), sm('y_test', 'init'),
                     sm('voxel_extract_1', 'v_origin', True), sm('X_train', 'init'), sm('y_train', 'init'),
                     sm('RF', 'modeling')): (
                            'X_test_data_0', 'y_test_data_10', 'RF_10'
                    ),
                    (sm('X_test', 'init'), sm('subgroup_1', 's_origin', True), sm('y_test', 'init'),
                     sm('voxel_extract_1', 'v_origin', True), sm('X_train', 'init'), sm('y_train', 'init'),
                     sm('RF', 'modeling')): (
                            'X_test_data_1', 'y_test_data_11', 'RF_11'
                    )
                },
        ),
        (
                # in_dicts
                [
                    {
                        (sm('y_test', 'init'), sm('voxel_extract_0', 'v_origin', True),
                         sm('subgroup_0', 's_origin', True)): 'y_test_data_00',
                        (sm('y_test', 'init'), sm('voxel_extract_0', 'v_origin', True),
                         sm('subgroup_1', 's_origin', True)): 'y_test_data_01',
                        (sm('y_test', 'init'), sm('voxel_extract_1', 'v_origin', True),
                         sm('subgroup_0', 's_origin', True)): 'y_test_data_10',
                        (sm('y_test', 'init'), sm('voxel_extract_1', 'v_origin', True),
                         sm('subgroup_1', 's_origin', True)): 'y_test_data_11',
                    },
                    {
                        (sm('X_train', 'init'), sm('y_train', 'init'), sm('voxel_extract_0', 'v_origin', True),
                         sm('subgroup_0', 's_origin', True), sm('RF', 'modeling')): 'RF_00',
                        (sm('X_train', 'init'), sm('y_train', 'init'), sm('voxel_extract_0', 'v_origin', True),
                         sm('subgroup_1', 's_origin', True), sm('RF', 'modeling')): 'RF_01',
                        (sm('X_train', 'init'), sm('y_train', 'init'), sm('voxel_extract_1', 'v_origin', True),
                         sm('subgroup_0', 's_origin', True), sm('RF', 'modeling')): 'RF_10',
                        (sm('X_train', 'init'), sm('y_train', 'init'), sm('voxel_extract_1', 'v_origin', True),
                         sm('subgroup_1', 's_origin', True), sm('RF', 'modeling')): 'RF_11',
                    },
                    {
                        (sm('X_test', 'init'), sm('subgroup_0', 's_origin', True)): 'X_test_data_0',
                        (sm('X_test', 'init'), sm('subgroup_1', 's_origin', True)): 'X_test_data_1'
                    },
                ],
                # out_dict
                {
                    (sm('y_test', 'init'), sm('voxel_extract_0', 'v_origin', True), sm('subgroup_0', 's_origin', True),
                     sm('X_train', 'init'), sm('y_train', 'init'), sm('RF', 'modeling'), sm('X_test', 'init')): (
                            'y_test_data_00', 'RF_00', 'X_test_data_0'
                    ),
                    (sm('y_test', 'init'), sm('voxel_extract_0', 'v_origin', True), sm('subgroup_1', 's_origin', True),
                     sm('X_train', 'init'), sm('y_train', 'init'), sm('RF', 'modeling'), sm('X_test', 'init')): (
                            'y_test_data_01', 'RF_01', 'X_test_data_1'
                    ),
                    (sm('y_test', 'init'), sm('voxel_extract_1', 'v_origin', True), sm('subgroup_0', 's_origin', True),
                     sm('X_train', 'init'), sm('y_train', 'init'), sm('RF', 'modeling'), sm('X_test', 'init')): (
                            'y_test_data_10', 'RF_10', 'X_test_data_0'
                    ),
                    (sm('y_test', 'init'), sm('voxel_extract_1', 'v_origin', True), sm('subgroup_1', 's_origin', True),
                     sm('X_train', 'init'), sm('y_train', 'init'), sm('RF', 'modeling'), sm('X_test', 'init')): (
                            'y_test_data_11', 'RF_11', 'X_test_data_1'
                    )
                },
        ),
        (
                # in_dicts
                [
                    {
                        (sm('y_test', 'init'), sm('voxel_extract_0', 'v_origin', True),
                         sm('subgroup_0', 's_origin', True)): 'y_test_data_00',
                        (sm('y_test', 'init'), sm('voxel_extract_0', 'v_origin', True),
                         sm('subgroup_1', 's_origin', True)): 'y_test_data_01',
                        (sm('y_test', 'init'), sm('voxel_extract_1', 'v_origin', True),
                         sm('subgroup_0', 's_origin', True)): 'y_test_data_10',
                        (sm('y_test', 'init'), sm('voxel_extract_1', 'v_origin', True),
                         sm('subgroup_1', 's_origin', True)): 'y_test_data_11',
                    },
                    {
                        (sm('X_test', 'init'), sm('subgroup_0', 's_origin', True)): 'X_test_data_0',
                        (sm('X_test', 'init'), sm('subgroup_1', 's_origin', True)): 'X_test_data_1'
                    },
                    {
                        (sm('X_train', 'init'), sm('y_train', 'init'), sm('voxel_extract_0', 'v_origin', True),
                         sm('subgroup_0', 's_origin', True), sm('RF', 'modeling')): 'RF_00',
                        (sm('X_train', 'init'), sm('y_train', 'init'), sm('voxel_extract_0', 'v_origin', True),
                         sm('subgroup_1', 's_origin', True), sm('RF', 'modeling')): 'RF_01',
                        (sm('X_train', 'init'), sm('y_train', 'init'), sm('voxel_extract_1', 'v_origin', True),
                         sm('subgroup_0', 's_origin', True), sm('RF', 'modeling')): 'RF_10',
                        (sm('X_train', 'init'), sm('y_train', 'init'), sm('voxel_extract_1', 'v_origin', True),
                         sm('subgroup_1', 's_origin', True), sm('RF', 'modeling')): 'RF_11',
                    },
                ],
                # out_dict
                {
                    (sm('y_test', 'init'), sm('voxel_extract_0', 'v_origin', True), sm('subgroup_0', 's_origin', True),
                     sm('X_test', 'init'), sm('X_train', 'init'), sm('y_train', 'init'), sm('RF', 'modeling')): (
                            'y_test_data_00', 'X_test_data_0', 'RF_00'
                    ),
                    (sm('y_test', 'init'), sm('voxel_extract_0', 'v_origin', True), sm('subgroup_1', 's_origin', True),
                     sm('X_test', 'init'), sm('X_train', 'init'), sm('y_train', 'init'), sm('RF', 'modeling')): (
                            'y_test_data_01', 'X_test_data_1', 'RF_01'
                    ),
                    (sm('y_test', 'init'), sm('voxel_extract_1', 'v_origin', True), sm('subgroup_0', 's_origin', True),
                     sm('X_test', 'init'), sm('X_train', 'init'), sm('y_train', 'init'), sm('RF', 'modeling')): (
                            'y_test_data_10', 'X_test_data_0', 'RF_10'
                    ),
                    (sm('y_test', 'init'), sm('voxel_extract_1', 'v_origin', True), sm('subgroup_1', 's_origin', True),
                     sm('X_test', 'init'), sm('X_train', 'init'), sm('y_train', 'init'), sm('RF', 'modeling')): (
                            'y_test_data_11', 'X_test_data_1', 'RF_11'
                    )
                },
        ),
        (
                # in_dicts
                [
                    {
                        (sm('X_test', 'init'), sm('feature_extraction_0', 'f_origin', True),
                         sm('subgroup_0', 's_origin', True)): 'X_test_data_00',
                        (sm('X_test', 'init'), sm('feature_extraction_0', 'f_origin', True),
                         sm('subgroup_1', 's_origin', True)): 'X_test_data_01',
                        (sm('X_test', 'init'), sm('feature_extraction_1', 'f_origin', True),
                         sm('subgroup_0', 's_origin', True)): 'X_test_data_10',
                        (sm('X_test', 'init'), sm('feature_extraction_1', 'f_origin', True),
                         sm('subgroup_1', 's_origin', True)): 'X_test_data_11',
                    },
                    {
                        (sm('y_test', 'init'), sm('voxel_extract_0', 'v_origin', True),
                         sm('subgroup_0', 's_origin', True)): 'y_test_data_00',
                        (sm('y_test', 'init'), sm('voxel_extract_0', 'v_origin', True),
                         sm('subgroup_1', 's_origin', True)): 'y_test_data_01',
                        (sm('y_test', 'init'), sm('voxel_extract_1', 'v_origin', True),
                         sm('subgroup_0', 's_origin', True)): 'y_test_data_10',
                        (sm('y_test', 'init'), sm('voxel_extract_1', 'v_origin', True),
                         sm('subgroup_1', 's_origin', True)): 'y_test_data_11',
                    },
                    {
                        (sm('X_train', 'init'), sm('feature_extraction_0', 'f_origin', True),
                         sm('subgroup_0', 's_origin', True),
                         sm('y_train', 'init'), sm('voxel_extract_0', 'v_origin', True),
                         sm('RF', 'modeling')): 'RF_000',
                        (sm('X_train', 'init'), sm('feature_extraction_0', 'f_origin', True),
                         sm('subgroup_0', 's_origin', True),
                         sm('y_train', 'init'), sm('voxel_extract_1', 'v_origin', True),
                         sm('RF', 'modeling')): 'RF_001',
                        (sm('X_train', 'init'), sm('feature_extraction_0', 'f_origin', True),
                         sm('subgroup_1', 's_origin', True),
                         sm('y_train', 'init'), sm('voxel_extract_0', 'v_origin', True),
                         sm('RF', 'modeling')): 'RF_010',
                        (sm('X_train', 'init'), sm('feature_extraction_0', 'f_origin', True),
                         sm('subgroup_1', 's_origin', True),
                         sm('y_train', 'init'), sm('voxel_extract_1', 'v_origin', True),
                         sm('RF', 'modeling')): 'RF_011',
                        (sm('X_train', 'init'), sm('feature_extraction_1', 'f_origin', True),
                         sm('subgroup_0', 's_origin', True),
                         sm('y_train', 'init'), sm('voxel_extract_0', 'v_origin', True),
                         sm('RF', 'modeling')): 'RF_100',
                        (sm('X_train', 'init'), sm('feature_extraction_1', 'f_origin', True),
                         sm('subgroup_0', 's_origin', True),
                         sm('y_train', 'init'), sm('voxel_extract_1', 'v_origin', True),
                         sm('RF', 'modeling')): 'RF_101',
                        (sm('X_train', 'init'), sm('feature_extraction_1', 'f_origin', True),
                         sm('subgroup_1', 's_origin', True),
                         sm('y_train', 'init'), sm('voxel_extract_0', 'v_origin', True),
                         sm('RF', 'modeling')): 'RF_110',
                        (sm('X_train', 'init'), sm('feature_extraction_1', 'f_origin', True),
                         sm('subgroup_1', 's_origin', True),
                         sm('y_train', 'init'), sm('voxel_extract_1', 'v_origin', True),
                         sm('RF', 'modeling')): 'RF_111',
                    },
                ],
                # out_dict
                {
                    (sm('X_test', 'init'), sm('feature_extraction_0', 'f_origin', True),
                     sm('subgroup_0', 's_origin', True),
                     sm('y_test', 'init'), sm('voxel_extract_0', 'v_origin', True),
                     sm('X_train', 'init'), sm('y_train', 'init'), sm('RF', 'modeling')): (
                            'X_test_data_00', 'y_test_data_00', 'RF_000'),
                    (sm('X_test', 'init'), sm('feature_extraction_0', 'f_origin', True),
                     sm('subgroup_0', 's_origin', True),
                     sm('y_test', 'init'), sm('voxel_extract_1', 'v_origin', True),
                     sm('X_train', 'init'), sm('y_train', 'init'), sm('RF', 'modeling')): (
                            'X_test_data_00', 'y_test_data_10', 'RF_001'),
                    (sm('X_test', 'init'), sm('feature_extraction_0', 'f_origin', True),
                     sm('subgroup_1', 's_origin', True),
                     sm('y_test', 'init'), sm('voxel_extract_0', 'v_origin', True),
                     sm('X_train', 'init'), sm('y_train', 'init'), sm('RF', 'modeling')): (
                            'X_test_data_01', 'y_test_data_01', 'RF_010'),
                    (sm('X_test', 'init'), sm('feature_extraction_0', 'f_origin', True),
                     sm('subgroup_1', 's_origin', True),
                     sm('y_test', 'init'), sm('voxel_extract_1', 'v_origin', True),
                     sm('X_train', 'init'), sm('y_train', 'init'), sm('RF', 'modeling')): (
                            'X_test_data_01', 'y_test_data_11', 'RF_011'),
                    (sm('X_test', 'init'), sm('feature_extraction_1', 'f_origin', True),
                     sm('subgroup_0', 's_origin', True),
                     sm('y_test', 'init'), sm('voxel_extract_0', 'v_origin', True),
                     sm('X_train', 'init'), sm('y_train', 'init'), sm('RF', 'modeling')): (
                            'X_test_data_10', 'y_test_data_00', 'RF_100'),
                    (sm('X_test', 'init'), sm('feature_extraction_1', 'f_origin', True),
                     sm('subgroup_0', 's_origin', True),
                     sm('y_test', 'init'), sm('voxel_extract_1', 'v_origin', True),
                     sm('X_train', 'init'), sm('y_train', 'init'), sm('RF', 'modeling')): (
                            'X_test_data_10', 'y_test_data_10', 'RF_101'),
                    (sm('X_test', 'init'), sm('feature_extraction_1', 'f_origin', True),
                     sm('subgroup_1', 's_origin', True),
                     sm('y_test', 'init'), sm('voxel_extract_0', 'v_origin', True),
                     sm('X_train', 'init'), sm('y_train', 'init'), sm('RF', 'modeling')): (
                            'X_test_data_11', 'y_test_data_01', 'RF_110'),
                    (sm('X_test', 'init'), sm('feature_extraction_1', 'f_origin', True),
                     sm('subgroup_1', 's_origin', True),
                     sm('y_test', 'init'), sm('voxel_extract_1', 'v_origin', True),
                     sm('X_train', 'init'), sm('y_train', 'init'), sm('RF', 'modeling')): (
                            'X_test_data_11', 'y_test_data_11', 'RF_111'),

                }
        ),
        (
                # in_dicts
                [
                    {
                        (sm('X_train', 'init'), sm('standardize_0', 's_origin', True)): 'X_train_0',
                        (sm('X_train', 'init'), sm('standardize_1', 's_origin', True)): 'X_train_1',
                    },
                    {(sm('y_train', 'init'),): 'y_train_data'}
                ],
                # out_dict
                {
                    (sm('X_train', 'init'), sm('standardize_0', 's_origin', True), sm('y_train', 'init')): (
                            'X_train_0', 'y_train_data'),
                    (sm('X_train', 'init'), sm('standardize_1', 's_origin', True), sm('y_train', 'init')): (
                            'X_train_1', 'y_train_data')
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
                        (sm('module_0', 'm_origin'),): lambda x, y: x + y,
                        (sm('module_1', 'm_origin'),): lambda x, y: x * y,
                    },
                    # data_dict
                    {(sm('data', 'init'),): [2, 3]}
                ],
                # out_dict
                {
                    (sm('data', 'init'), sm('module_0', 'm_origin')): 5,
                    (sm('data', 'init'), sm('module_1', 'm_origin')): 6
                }
        ),
        (
                # in_dicts
                [
                    # modules
                    {
                        (sm('group_0', 'g_origin', True), sm('module_0', 'm_origin'),): lambda x, y: x + y,
                        (sm('group_1', 'g_origin', True), sm('module_1', 'm_origin'),): lambda x, y: x * y,
                    },
                    # data_dict
                    {
                        (sm('data', 'init'), sm('group_0', 'g_origin', True)): [np.array([1, 2, 3]),
                                                                                np.array([4, 5, 6])],
                        (sm('data', 'init'), sm('group_1', 'g_origin', True)): [np.array([1, 2, 3]),
                                                                                np.array([4, 5, 6])],
                    }
                ],
                # out_dict
                {
                    (sm('data', 'init'), sm('group_0', 'g_origin', True), sm('module_0', 'm_origin')): np.array(
                        [5, 7, 9]),
                    (sm('data', 'init'), sm('group_1', 'g_origin', True), sm('module_1', 'm_origin')): np.array(
                        [4, 10, 18]),
                }
        ),
        (
                # in_dicts
                [
                    # modules
                    {
                        (sm('data', 'init'), sm('group_0', 'g_origin', True), sm('module_0', 'm_origin'),): lambda x,
                                                                                                                   y: x + y,
                        (sm('data', 'init'), sm('group_1', 'g_origin', True), sm('module_1', 'm_origin'),): lambda x,
                                                                                                                   y: x * y,
                    },
                    # data_dict
                    {
                        (sm('data', 'init2'), sm('group_0', 'g_origin', True)): [np.array([1, 2, 3]),
                                                                                 np.array([4, 5, 6])],
                        (sm('data', 'init2'), sm('group_1', 'g_origin', True)): [np.array([1, 2, 3]),
                                                                                 np.array([4, 5, 6])],
                    }
                ],
                # out_dict
                {
                    (sm('data', 'init2'), sm('group_0', 'g_origin', True), sm('data', 'init'),
                     sm('module_0', 'm_origin')): np.array([5, 7, 9]),
                    (sm('data', 'init2'), sm('group_1', 'g_origin', True), sm('data', 'init'),
                     sm('module_1', 'm_origin')): np.array([4, 10, 18]),
                },
        ),
    ]
)

class TestApplyModules:

    def test_apply_modules(self, in_dicts, out_dict):
        result_dict = apply_modules(*in_dicts)
        assert_equal(result_dict, out_dict)

class TestConvert:

    def test_dict_to_df(self):
        in_dict_1 = {(sm('X_train', 'init'), sm('feat_extract_0', 'feat_extract'), 
            sm('y_train', 'init'), sm('DT', 'modeling'), sm('acc', 'metrics')): 0.9,
            (sm('X_train', 'init'), sm('feat_extract_1', 'feat_extract'), 
            sm('y_train', 'init'), sm('DT', 'modeling'), sm('acc', 'metrics')): 0.95}
        out_df_1 = pd.DataFrame(data={'init-feat_extract': ['X_train', 'X_train'],
                                'feat_extract': ['feat_extract_0', 'feat_extract_1'],
                                'init-modeling': ['y_train', 'y_train'],
                                'modeling': ['DT', 'DT'],
                                'metrics': ['acc', 'acc'],
                                'out': [0.9, 0.95]})
        in_dict_2 = {(sm('X_train', 'init'), sm('sample_0', 'sample'), sm('y_train', 'init'), 
                    sm(('k=10', 'e=1e-3'), 'modeling'), sm('s_0', 'stability')): 0.333,
                    (sm('X_train', 'init'), sm('sample_0', 'sample'), sm('y_train', 'init'), 
                    sm(('k=10', 'e=1e-5'), 'modeling'), sm('s_0', 'stability')): 0.452}
        out_df_2 = pd.DataFrame(data={'init-sample': ['X_train', 'X_train'],
                                'sample': ['sample_0', 'sample_0'],
                                'init-modeling': ['y_train', 'y_train'],
                                'k-modeling': ['10', '10'],
                                'e-modeling': ['1e-3', '1e-5'],
                                'stability': ['s_0', 's_0'],
                                'out': [0.333, 0.452]})
        assert dict_to_df(in_dict_1).equals(out_df_1)
        assert dict_to_df(in_dict_2, param_key='modeling').equals(out_df_2)

    def test_compute_interval(self):
        in_dict = {(sm('X_train', 'init'), sm('feat_extract_0', 'feat_extract'), 
            sm('y_train', 'init'), sm('DT', 'modeling'), sm('acc', 'metrics')): 0.9,
            (sm('X_train', 'init'), sm('feat_extract_1', 'feat_extract'), 
            sm('y_train', 'init'), sm('DT', 'modeling'), sm('acc', 'metrics')): 0.95}
        df = dict_to_df(in_dict)
        interval = compute_interval(df, 'out', 'metrics')
        assert interval['out']['std'][0] == 0.03535533905932729
