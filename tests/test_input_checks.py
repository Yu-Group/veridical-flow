"""Invalid input raises a clear ValueError or TypeError (these were asserts,
which disappear under python -O)."""

import numpy as np
import pytest

from vflow import AsyncVfunc, Vfunc, Vset, build_vset, cum_acc_by_uncertainty, init_args


def test_init_args_names_must_match():
    with pytest.raises(ValueError, match="same length"):
        init_args([1, 2], names=["only_one"])


def test_build_vset_checks():
    with pytest.raises(ValueError, match="same length"):
        build_vset("v", [np.sum, np.mean], param_dict=[{}, {}, {}])
    with pytest.raises(TypeError, match="callable"):
        build_vset("v", "not a function")


def test_vset_checks():
    with pytest.raises(TypeError, match="not a list"):
        Vset("v", [np.sum], vfunc_keys=("s",))
    with pytest.raises(ValueError, match="same length"):
        Vset("v", [np.sum, np.mean], vfunc_keys=["s"])
    with pytest.raises(TypeError, match="init_args"):
        Vset("v", [np.sum])(np.ones(3))
    with pytest.raises(AttributeError, match="fit the Vset"):
        Vset("v", [np.sum]).transform(*init_args([np.ones(3)]))


@pytest.mark.parametrize("cls", [Vfunc, AsyncVfunc])
def test_vfunc_must_be_callable(cls):
    with pytest.raises(TypeError, match="callable"):
        cls("v", 3)


def test_vset_container_methods():
    vset = Vset("v", [np.sum, np.mean], vfunc_keys=["s", "m"])
    key = next(iter(vset.keys()))
    assert key in vset and ("nope",) not in vset
    assert vset[key].vfunc is np.sum and len(vset) == 2


def test_pipeline_len():
    from vflow import PCSPipeline

    assert len(PCSPipeline([Vset("v", [np.sum])])) == 1


def test_filter_needs_the_vset_in_the_results():
    from vflow import filter_vset_by_metric

    (x,) = init_args([np.ones(3)], names=["x"])
    scored = Vset("score", [np.sum], vfunc_keys=["s"])
    with pytest.raises(ValueError, match="should be one of the columns"):
        filter_vset_by_metric(scored(x), Vset("other", [np.mean]))


def test_predict_proba_uncertainty_without_group_by():
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(0)
    X, y = init_args([rng.normal(size=(20, 2)), [0, 1] * 10], names=["X", "y"])
    models = Vset("m", [LogisticRegression(), LogisticRegression(C=0.1)])
    models.fit(X, y)
    mean, std, _ = models.predict_proba(X, with_uncertainty=True)
    assert len([k for k in mean if k != "__prev__"]) == 1


def test_async_transform():
    from sklearn.preprocessing import StandardScaler

    (X,) = init_args([np.arange(6.0).reshape(3, 2)], names=["X"])
    scaler = Vset("scale", [StandardScaler()], vfunc_keys=["std"], is_async=True)
    (out,) = [v for k, v in scaler.fit(X).transform(X).items() if k != "__prev__"]
    assert np.allclose(out.mean(axis=0), 0)


def test_cum_acc_by_uncertainty_checks():
    probs = {("a",): np.full((4, 2), 0.5)}
    with pytest.raises(ValueError, match="same keys"):
        cum_acc_by_uncertainty(probs, {("b",): np.full((4, 2), 0.1)}, [0, 1, 0, 1])
    with pytest.raises(ValueError, match="same as predictions"):
        cum_acc_by_uncertainty(probs, probs, [0, 1])
    (labels,) = init_args([np.array([0, 1, 0, 1])])
    with pytest.raises(ValueError, match="single 1D vector"):
        cum_acc_by_uncertainty(probs, probs, {**labels, ("x",): [1, 0, 1, 0]})
    uncertainty, cum_acc, _ = cum_acc_by_uncertainty(probs, probs, labels)
    assert uncertainty.shape == cum_acc.shape == (1, 4)
