from functools import partial

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample

from vflow import build_vset, init_args


def draws(vset, X):
    """Run every vfunc of `vset` on X and return the outputs in vfunc order."""
    (Xs,) = [vset(*init_args([X], names=["X"]), n_out=1)]
    return [v for k, v in Xs.items() if k != "__prev__"]


class TestSeeding:
    def setup_method(self):
        self.X = np.arange(40).reshape(20, 2)

    def test_reps_differ_and_reproduce(self):
        a = draws(build_vset("boot", resample, reps=4, seed=0), self.X)
        assert len(a) == 4
        assert len({x.tobytes() for x in a}) == 4, "each rep draws its own sample"
        b = draws(build_vset("boot", resample, reps=4, seed=0), self.X)
        assert all(np.array_equal(x, y) for x, y in zip(a, b)), "same seed, same draws"
        c = draws(build_vset("boot", resample, reps=4, seed=1), self.X)
        assert not all(np.array_equal(x, y) for x, y in zip(a, c))

    def test_seed_with_param_dict_and_class(self):
        X, y = make_classification(n_samples=60, n_features=4, random_state=0)
        vset = build_vset(
            "RF",
            RandomForestClassifier,
            param_dict={"max_depth": [2, 3]},
            reps=2,
            seed=7,
            n_estimators=5,
        )
        states = [vf.vfunc.random_state for vf in vset.vfuncs.values()]
        assert len(states) == 4 and len(set(states)) == 4
        again = build_vset(
            "RF",
            RandomForestClassifier,
            param_dict={"max_depth": [2, 3]},
            reps=2,
            seed=7,
            n_estimators=5,
        )
        assert states == [vf.vfunc.random_state for vf in again.vfuncs.values()]
        p1 = [vf.vfunc.fit(X, y).predict_proba(X) for vf in vset.vfuncs.values()]
        p2 = [vf.vfunc.fit(X, y).predict_proba(X) for vf in again.vfuncs.values()]
        assert all(np.array_equal(u, v) for u, v in zip(p1, p2))

    def test_no_seed_is_unchanged(self):
        vset = build_vset("boot", resample, reps=3)
        assert all(
            "random_state" not in vf.vfunc.keywords for vf in vset.vfuncs.values()
        )

    def test_seed_conflicts_and_unseedable_funcs(self):
        with pytest.raises(ValueError, match="either `seed` or `random_state`"):
            build_vset("boot", resample, reps=2, seed=0, random_state=3)
        with pytest.raises(ValueError, match="either `seed` or `random_state`"):
            build_vset("boot", resample, param_dict={"random_state": [1, 2]}, seed=0)

        def double(X):
            return 2 * X

        with pytest.raises(ValueError, match="does not"):
            build_vset("double", double, reps=2, seed=0)

    def test_partial_and_callable_object_names(self):
        vset = build_vset(
            "boot", partial(resample, replace=True), param_dict={"n_samples": [5]}
        )
        assert next(iter(vset.vfuncs))[0].value[0] == "func=resample(replace=True)"

        class Scale:
            def __call__(self, X, factor=1):
                return factor * X

        vset = build_vset("scale", Scale(), param_dict={"factor": [2, 3]})
        names = [k[0].value[0] for k in vset.vfuncs]
        assert names == ["func=Scale", "func=Scale"]
