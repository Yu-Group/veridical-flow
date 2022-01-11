from numpy.testing import assert_equal

from vflow.helpers import *
from vflow.subkey import Subkey as sm

class TestHelpers:

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
        in_dict_3 = {(sm('X_train', 'init'), sm('sample_0', 'sample'), sm('y_train', 'init'),
                    sm(('k=10', 'e=1e-3'), 'modeling'), sm('s_0', 'stability')): [0.333, 0.222],
                    (sm('X_train', 'init'), sm('sample_0', 'sample'), sm('y_train', 'init'),
                     sm(('k=10', 'e=1e-5'), 'modeling'), sm('s_0', 'stability')): [0.452, 0.322]}
        out_df_3 = pd.DataFrame(data={'init-sample': ['X_train', 'X_train'],
                                'sample': ['sample_0', 'sample_0'],
                                'init-modeling': ['y_train', 'y_train'],
                                'modeling': [('k=10', 'e=1e-3'), ('k=10', 'e=1e-5')],
                                'stability': ['s_0', 's_0'],
                                'out': [[0.333, 0.222], [0.452, 0.322]],
                                'out-0': [0.333, 0.452],
                                'out-1': [0.222, 0.322]})
        in_dict_4 = {(sm('X_train', 'init'), sm('sample_0', 'sample'), sm('y_train', 'init'),
                    sm(('k=10', 'e=1e-3'), 'modeling'), sm('s_0', 'stability')): {'k1': 0.333, 'k2': 0.222},
                    (sm('X_train', 'init'), sm('sample_0', 'sample'), sm('y_train', 'init'),
                     sm(('k=10', 'e=1e-5'), 'modeling'), sm('s_0', 'stability')): {'k1': 0.452, 'k2': 0.322}}
        out_df_4 = pd.DataFrame(data={'init-sample': ['X_train', 'X_train'],
                                'sample': ['sample_0', 'sample_0'],
                                'init-modeling': ['y_train', 'y_train'],
                                'modeling': [('k=10', 'e=1e-3'), ('k=10', 'e=1e-5')],
                                'stability': ['s_0', 's_0'],
                                'out': [{'k1': 0.333, 'k2': 0.222}, {'k1': 0.452, 'k2': 0.322}],
                                'out-k1': [0.333, 0.452],
                                'out-k2': [0.222, 0.322]})
        assert dict_to_df(in_dict_1).equals(out_df_1)
        assert dict_to_df(in_dict_2, param_key='modeling').equals(out_df_2)
        assert dict_to_df(in_dict_3, param_key='out').equals(out_df_3)
        assert dict_to_df(in_dict_4, param_key='out').equals(out_df_4)
    
    def test_perturbation_stats(self):
        in_dict = {(sm('X_train', 'init'), sm('feat_extract_0', 'feat_extract'), 
            sm('y_train', 'init'), sm('DT', 'modeling'), sm('feat_imp', 'metrics')): 0.455,
            (sm('X_train', 'init'), sm('feat_extract_0', 'feat_extract'), 
            sm('y_train', 'init'), sm('LR', 'modeling'), sm('feat_imp', 'metrics')): 0.522,
            (sm('X_train', 'init'), sm('feat_extract_1', 'feat_extract'), 
            sm('y_train', 'init'), sm('DT', 'modeling'), sm('feat_imp', 'metrics')): 0.76,
            (sm('X_train', 'init'), sm('feat_extract_1', 'feat_extract'), 
            sm('y_train', 'init'), sm('LR', 'modeling'), sm('feat_imp', 'metrics')): 0.95}
        df = dict_to_df(in_dict)
        stats = perturbation_stats(df, 'feat_extract')
        cols = ['feat_extract', 'count', 'mean', 'std']
        assert all(c in cols for c in stats.columns)
        assert round(stats.loc[0]['mean'], 4) == 0.4885
        assert round(stats.loc[1]['std'], 6) == 0.134350

        in_dict = {(sm('X_train', 'init'), sm('feat_extract_0', 'feat_extract'), 
            sm('y_train', 'init'), sm('DT', 'modeling'), sm('feat_imp', 'metrics')): [0.6, 0.3, 0.4],
            (sm('X_train', 'init'), sm('feat_extract_0', 'feat_extract'), 
            sm('y_train', 'init'), sm('LR', 'modeling'), sm('feat_imp', 'metrics')): [0.94, 0.33, 0.24],
            (sm('X_train', 'init'), sm('feat_extract_1', 'feat_extract'), 
            sm('y_train', 'init'), sm('DT', 'modeling'), sm('feat_imp', 'metrics')): [0.26, 0.31, 0.47],
            (sm('X_train', 'init'), sm('feat_extract_1', 'feat_extract'), 
            sm('y_train', 'init'), sm('LR', 'modeling'), sm('feat_imp', 'metrics')): [0.76, 0.883, 0.354]}
        df = dict_to_df(in_dict)
        stats = perturbation_stats(df, 'feat_extract', prefix='o', split=True)
        assert len(stats.columns) == 8
        assert stats.columns[1] == 'o-count'
        assert stats.columns[-1] == 'o2-std'
        assert stats.loc[1]['o2-std'] == 0.0820243866176395

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