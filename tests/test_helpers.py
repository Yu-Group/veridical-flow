from numpy.testing import assert_equal

from vflow.helpers import *

class TestHelpers:

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