"""Regression tests for bugs found by review, one test per bug."""

import numpy as np
import pandas as pd
import pytest

from vflow import (
    PCSPipeline,
    Vset,
    build_vset,
    cum_acc_by_uncertainty,
    dict_to_df,
    filter_vset_by_metric,
    init_args,
    perturbation_stats,
)


def values(d):
    """The outputs in a Vset output dict, without its bookkeeping entry."""
    return [v for k, v in d.items() if k != "__prev__"]


def model(X, a=1):
    """A 'model' whose score is just its parameter, so the best one is known."""
    return float(a) + 0 * np.sum(X)


class TestFilterVsetByMetric:
    def setup_method(self):
        (self.X,) = init_args([np.ones((4, 2))], names=["X"])

    def test_vsets_without_param_dict(self):
        def low(X):
            return 1.0

        def high(X):
            return 2.0

        vset = build_vset("plain", [low, high])
        kept = filter_vset_by_metric(vset(self.X), vset, n_keep=1)
        assert len(kept.vfuncs) == 1, "used to keep nothing"
        assert next(iter(kept.vfuncs.values())).vfunc(None) == 2.0

    def test_vsets_with_reps(self):
        vset = build_vset("m", model, param_dict={"a": [1, 3]}, reps=2)
        kept = filter_vset_by_metric(vset(self.X), vset, n_keep=1)
        assert len(kept.vfuncs) == 1, "used to keep nothing when reps > 1"

    def test_n_keep_counts_distinct_vfuncs(self):
        boot = Vset("boot", [lambda X: X, lambda X: 2 * X], vfunc_keys=["b0", "b1"])
        mset = build_vset("m", model, param_dict={"a": [1, 2, 3]})
        scores = mset(boot(self.X))
        kept = filter_vset_by_metric(scores, mset, n_keep=2)
        # a=3 tops both bootstrap rows; it must count once
        assert len(kept.vfuncs) == 2
        assert {k[0].value for k in kept.vfuncs} == {
            ("func=model", "a=3"),
            ("func=model", "a=2"),
        }

    def test_filtering_a_filtered_vset(self):
        boot = Vset("boot", [lambda X: X], vfunc_keys=["b0"])
        mset = build_vset("m", model, param_dict={"a": [1, 2, 3]})
        subs = boot(self.X)
        once = filter_vset_by_metric(mset(subs), mset, n_keep=2)
        twice = filter_vset_by_metric(once(subs), once, n_keep=1)
        assert [k[0].value for k in twice.vfuncs] == [("func=model", "a=3")]


class TestPerturbationStats:
    def setup_method(self):
        self.df = pd.DataFrame({"g": ["a", "a", "b", "b"], "out": [1.0, 3.0, 2.0, 6.0]})

    def test_callable_and_string_func(self):
        by_mean = perturbation_stats(self.df, "g", func=np.mean)
        assert list(by_mean["mean"]) == [2.0, 4.0]
        by_std = perturbation_stats(self.df, "g", func="std")
        assert list(by_std.columns) == ["g", "std"]

    def test_default_columns_unchanged_and_prefix_honoured(self):
        assert list(perturbation_stats(self.df, "g").columns) == [
            "g",
            "count",
            "mean",
            "std",
        ]
        cols = perturbation_stats(self.df, "g", prefix="acc").columns
        assert list(cols) == ["g", "acc-count", "acc-mean", "acc-std"]

    def test_array_outputs_without_count(self):
        df = pd.DataFrame(
            {"g": ["a", "a"], "out": [np.array([1.0, 2.0]), np.array([3.0, 4.0])]}
        )
        out = perturbation_stats(df, "g", func=["mean"])
        assert "out-count" not in out.columns
        assert np.allclose(out["out-mean"].iloc[0], [2.0, 3.0])


class TestDictToDf:
    def setup_method(self):
        (self.X,) = init_args([np.ones(3)], names=["X"])

    def test_param_key_with_different_param_names(self):
        def f(X, a=0):
            return a

        def g(X, b=0):
            return -b

        vset = build_vset("v", [f, g], param_dict=[{"a": [1]}, {"b": [2]}])
        df = dict_to_df(vset(self.X), param_key="v")
        row_g = df[df["func-v"] == "g"].iloc[0]
        assert row_g["b-v"] == "2" and pd.isna(row_g["a-v"])
        assert list(df.columns) == ["init-v", "func-v", "a-v", "b-v", "out"]

    def test_param_values_containing_equals(self):
        def f(X, q="x"):
            return q

        vset = build_vset("v", f, param_dict={"q": ["k=v"]})
        df = dict_to_df(vset(self.X), param_key="v")
        assert df["q-v"].iloc[0] == "k=v"

    def test_plain_string_keys_give_a_clear_error(self):
        vset = Vset("v", [np.sum], vfunc_keys=["s"])
        with pytest.raises(ValueError, match="name=value"):
            dict_to_df(vset(self.X), param_key="v")


def test_cum_acc_by_uncertainty_rejects_class_predictions():
    preds = {("m",): np.array([0, 1, 1])}
    with pytest.raises(ValueError, match="class probabilities"):
        cum_acc_by_uncertainty(preds, preds, [0, 1, 1])


class TestPCSPipeline:
    def test_generate_names(self):
        p = PCSPipeline(
            [
                Vset("a", [np.abs, np.square], vfunc_keys=["abs", "sq"]),
                Vset("b", [np.sum], vfunc_keys=["sum"]),
            ]
        )
        names = p.generate_names()
        assert list(names.columns) == ["a", "b"] and len(names) == 2
        flat = p.generate_names(as_pandas=False)
        assert len(flat) == 2 and flat[0][0].startswith("a_0_")

    def test_run_chains_steps_and_returns_outputs(self):
        p = PCSPipeline(
            [
                Vset("a", [np.abs, np.square], vfunc_keys=["abs", "sq"]),
                Vset("b", [np.sum], vfunc_keys=["sum"]),
            ]
        )
        (x,) = init_args([np.array([-1.0, 2.0])], names=["x"])
        out = p.run(x)
        assert sorted(values(out)) == [3.0, 5.0]


class TestVsetConstruction:
    def test_dict_with_plain_keys(self):
        from sklearn.linear_model import LogisticRegression

        vset = Vset("m", {"LR": LogisticRegression()})
        X, y = init_args([np.random.default_rng(0).normal(size=(20, 2)), [0, 1] * 10])
        vset.fit(X, y)
        assert [k[0].value for k in vset.vfuncs] == ["LR"]
        assert next(iter(vset.vfuncs))[0].origin == "m"

    def test_duplicate_keys_and_bad_containers(self):
        with pytest.raises(ValueError, match="unique"):
            Vset("m", [np.sum, np.mean], vfunc_keys=["s", "s"])
        with pytest.raises(TypeError, match="list or a dict"):
            Vset("m", (np.sum, np.mean))

    def test_build_vset_merges_identical_configurations(self):
        def f(X, a=0):
            return a

        vset = build_vset("v", f, param_dict=[{"a": [1, 2]}, {"a": [2, 3]}])
        assert len(vset) == 3

    def test_predict_needs_a_predictor(self):
        (x,) = init_args([np.ones(3)], names=["x"])
        vset = Vset("s", [np.sum], vfunc_keys=["s"]).fit(x)
        with pytest.raises(AttributeError, match="'predict'"):
            vset.predict(x)


def test_subkey_hash_matches_eq():
    from vflow import Subkey

    a, b = Subkey("x", "o", True), Subkey("x", "o", False)
    assert a == b and hash(a) == hash(b)
    assert (a,) in {(b,): 1}


class TestMlflowTracking:
    def test_params_logged_and_runs_closed(self, tmp_path):
        from mlflow.tracking import MlflowClient

        (x,) = init_args([np.array([-1.0, 2.0])], names=["x"])
        up = Vset("up", [np.abs, np.square], vfunc_keys=["abs", "sq"])
        met = Vset("met", [np.sum], vfunc_keys=["sum"], tracking_dir=str(tmp_path))
        met(up(x))
        client = MlflowClient(tracking_uri=str(tmp_path))
        runs = client.search_runs([met._exp_id])
        assert len(runs) == 2
        assert sorted(r.data.params["up"] for r in runs) == ["abs", "sq"]
        assert {r.info.status for r in runs} == {"FINISHED"}

    def test_fit_with_tracking_does_not_crash(self, tmp_path):
        from sklearn.linear_model import LogisticRegression

        X, y = init_args([np.random.default_rng(0).normal(size=(20, 2)), [0, 1] * 10])
        Vset(
            "mm", [LogisticRegression()], vfunc_keys=["LR"], tracking_dir=str(tmp_path)
        ).fit(X, y)


class TestCaching:
    def test_redefined_function_is_not_served_stale(self, tmp_path):
        (x,) = init_args([np.array([1.0, 2.0])], names=["x"])

        def f(a):
            return a + 1

        first = values(Vset("s", [f], vfunc_keys=["f"], cache_dir=str(tmp_path))(x))

        def f(a):  # noqa: F811 - redefined on purpose, as in a notebook
            return a + 100

        second = values(Vset("s", [f], vfunc_keys=["f"], cache_dir=str(tmp_path))(x))
        assert np.array_equal(first[0], [2.0, 3.0])
        assert np.array_equal(second[0], [101.0, 102.0])

    def test_lambdas_and_cache_hits(self, tmp_path):
        (x,) = init_args([np.array([1.0, 2.0])], names=["x"])
        # count calls in a file: a closure over a mutable list would change the
        # function's fingerprint after the first call
        log = str(tmp_path / "calls.txt")

        def slow(a):
            with open(log, "a") as f:
                f.write("x")
            return a * 2

        vset = Vset("lam", [lambda a: a * 2, slow], cache_dir=str(tmp_path / "c"))
        out1, out2 = values(vset(x)), values(vset(x))
        assert all(np.array_equal(u, v) for u, v in zip(out1, out2))
        with open(log) as f:
            assert f.read() == "x", "the second call is served from the cache"


def test_lazy_async_feeds_lazy_sync():
    (x,) = init_args([np.array([1.0, 2.0])], names=["x"])
    up = Vset("up", [lambda a: a + 1], is_async=True, lazy=True)
    down = Vset("down", [lambda a: a * 2], lazy=True)
    (promise,) = values(down(up(x)))
    assert np.array_equal(promise(), [4.0, 6.0])


def test_async_detected_from_dict_of_async_vfuncs():
    from vflow import AsyncVfunc, Subkey

    (x,) = init_args([np.array([1.0, 2.0])], names=["x"])
    vset = Vset("s", {(Subkey("s", "s"),): AsyncVfunc("s", np.sum)})
    assert vset._async
    assert values(vset(x)) == [3.0]


class TestArrayPerturbationStats:
    def setup_method(self):
        # two 3x2 "predict_proba" outputs in one group
        self.df = pd.DataFrame(
            {"g": ["a", "a"], "out": [np.zeros((3, 2)), np.ones((3, 2))]}
        )

    def test_split_flattens_2d_entries(self):
        out = perturbation_stats(self.df, "g", split=True)
        assert "out2_1-mean" in out.columns and "out0_0-std" in out.columns
        assert out["out2_1-mean"].iloc[0] == 0.5

    def test_tuple_group_keys(self):
        df = self.df.assign(g=[("f", "C=1"), ("f", "C=1")])
        out = perturbation_stats(df, "g")
        assert len(out) == 1 and out["out-mean"].iloc[0].shape == (3, 2)

    def test_predict_proba_with_uncertainty_grouped_by_model(self):
        from sklearn.linear_model import LogisticRegression

        rng = np.random.default_rng(0)
        X, y = init_args([rng.normal(size=(20, 2)), [0, 1] * 10], names=["X", "y"])
        boot = build_vset("boot", sklearn_resample, reps=2, seed=0)
        Xb, yb = boot(X, y)
        models = build_vset("model", LogisticRegression, param_dict={"C": [0.1, 1]})
        models.fit(Xb, yb)
        mean, std, _ = models.predict_proba(
            X, with_uncertainty=True, group_by=["model"]
        )
        assert len(values(mean)) == 2

    def test_unsupported_funcs_raise(self):
        with pytest.raises(ValueError, match="supports only"):
            perturbation_stats(self.df, "g", func=["mean", "max"])


def sklearn_resample(*arrays, random_state=None):
    from sklearn.utils import resample

    return resample(*arrays, random_state=random_state)


class TestBuildVsetNames:
    def test_partials_of_one_function_are_kept_apart(self):
        from functools import partial

        def draw(X, n=1, random_state=None):
            return n

        vset = build_vset(
            "d", [partial(draw, n=5), partial(draw, n=10)], reps=2, seed=0
        )
        assert len(vset) == 4

    def test_distinct_configs_with_one_key_raise(self):
        with pytest.raises(ValueError, match="share the key"):
            build_vset(
                "m", [lambda x, a: x + a, lambda x, a: x * a], param_dict={"a": [1]}
            )
        with pytest.raises(ValueError, match="share the key"):
            build_vset("n", model, param_dict={"a": [1, "1"]})


class TestCacheKeys:
    def test_equal_copy_of_inputs_hits_the_cache(self, tmp_path):
        import cloudpickle

        log = str(tmp_path / "calls.txt")

        def stage2(a):
            with open(log, "a") as f:
                f.write("x")
            return a * 2

        (x,) = init_args([np.array([1.0, 2.0])], names=["x"])
        out1 = Vset("s1", [lambda a: a + 1], vfunc_keys=["p"])(x)
        v2 = Vset("s2", [stage2], vfunc_keys=["d"], cache_dir=str(tmp_path / "c"))
        v2(out1)
        # an output loaded back from a cache is an equal copy, not the object
        v2(cloudpickle.loads(cloudpickle.dumps(out1)))
        with open(log) as f:
            assert f.read() == "x"

    def test_edited_module_function_is_not_served_stale(self, tmp_path, monkeypatch):
        import importlib

        mod = tmp_path / "vflow_test_mod.py"
        mod.write_text("def f(x):\n    return float(x.sum())\n")
        monkeypatch.syspath_prepend(str(tmp_path))
        m = importlib.import_module("vflow_test_mod")
        (x,) = init_args([np.array([1.0, 2.0])], names=["x"])
        cache = str(tmp_path / "c")
        assert values(Vset("s", [m.f], vfunc_keys=["f"], cache_dir=cache)(x)) == [3.0]
        mod.write_text("def f(x):\n    return float(x.sum()) * 100\n")
        m = importlib.reload(m)
        out = values(Vset("s", [m.f], vfunc_keys=["f"], cache_dir=cache)(x))
        assert out == [300.0]


def test_mlflow_metric_names_from_build_vset_keys(tmp_path):
    from mlflow.tracking import MlflowClient

    def score(x, scale=1):
        return float(np.sum(x)) * scale

    (x,) = init_args([np.array([1.0, 2.0])], names=["x"])
    met = build_vset(
        "met", score, param_dict={"scale": [1, 2]}, tracking_dir=str(tmp_path)
    )
    met(x)
    (run,) = MlflowClient(tracking_uri=str(tmp_path)).search_runs([met._exp_id])
    assert run.data.metrics == {"func_score_scale_1": 3.0, "func_score_scale_2": 6.0}
    assert run.info.status == "FINISHED"


def test_pipeline_run_with_a_fitted_step():
    a = Vset("a", [np.abs], vfunc_keys=["abs"])
    b = Vset("b", [np.sum], vfunc_keys=["sum"])
    (x,) = init_args([np.array([-1.0, 2.0])], names=["x"])
    a.fit(x)
    assert values(PCSPipeline([a, b]).run(x)) == [3.0]


def test_vfunc_keys_that_compare_equal_are_duplicates():
    with pytest.raises(ValueError, match="unique"):
        Vset("m", [np.sum, np.mean], vfunc_keys=[1, 1.0])
