import numpy as np
import pandas as pd
import pytest

from vflow import Vset, build_vset, init_args, stability_report
from vflow.report import OTHER


def results(scores):
    """A DataFrame shaped like dict_to_df output: boot x model x metric."""
    rows = [
        {"init-boot": "X", "boot": b, "model": m, "metric": "acc", "out": s}
        for (b, m), s in scores.items()
    ]
    return pd.DataFrame(rows)


def share(report, vset):
    v = report.variance
    return float(v.loc[v["vset"] == vset, "share"].iloc[0])


def test_variance_shares_for_a_pure_model_effect():
    # the model shifts the score; the bootstrap sample never does
    df = results(
        {(b, m): {"LR": 0.7, "RF": 0.9}[m] for b in "abc" for m in ["LR", "RF"]}
    )
    report = stability_report(df)
    assert share(report, "model") == pytest.approx(1.0)
    assert share(report, "boot") == pytest.approx(0.0)
    assert share(report, OTHER) == pytest.approx(0.0)
    assert "most of the variation comes from `model` (100%" in " ".join(
        report.headlines
    )


def test_shares_add_up_and_interactions_are_the_rest():
    # model and boot interact: which model wins depends on the sample
    df = results({("a", "LR"): 1, ("a", "RF"): 0, ("b", "LR"): 0, ("b", "RF"): 1})
    report = stability_report(df)
    assert share(report, "model") == pytest.approx(0.0)
    assert share(report, OTHER) == pytest.approx(1.0)
    assert any("no single Vset" in h for h in report.headlines)


def test_ranking_counts_wins_across_the_other_perturbations():
    scores = {("a", "LR"): 0.8, ("a", "RF"): 0.9, ("b", "LR"): 0.85, ("b", "RF"): 0.7}
    scores.update({("c", "LR"): 0.6, ("c", "RF"): 0.95})
    report = stability_report(results(scores), compare="model")
    r = report.ranking.set_index("model")
    assert r.loc["RF", "wins"] == 2 and r.loc["LR", "wins"] == 1
    assert r.loc["RF", "win_rate"] == pytest.approx(2 / 3)
    assert report.ranking["model"].iloc[0] == "RF"
    assert any("`RF` is best in 2 of 3" in h for h in report.headlines)

    lower = stability_report(results(scores), compare="model", bigger_is_better=False)
    assert lower.ranking["model"].iloc[0] == "LR"


def test_by_reports_each_metric_separately():
    df = results({(b, m): 0.5 for b in "ab" for m in ["LR", "RF"]})
    other = df.assign(metric="auc", out=[0.6, 0.9, 0.6, 0.9])
    df = pd.concat([df, other], ignore_index=True)

    auto = stability_report(df, compare="model")
    assert auto.by == "metric"
    assert list(auto.summary["metric"]) == ["acc", "auc"]
    assert set(auto.ranking["metric"]) == {"acc", "auc"}
    assert any("`metric` = acc: the results do not vary" in h for h in auto.headlines)

    pooled = stability_report(df, by=None)
    assert pooled.by is None and len(pooled.summary) == 1
    assert "metric" in set(pooled.variance["vset"])


def test_end_to_end_from_a_pipeline_output_dict():
    rng = np.random.default_rng(0)
    X, y = init_args([rng.normal(size=(30, 2)), np.r_[[0] * 15, [1] * 15]])
    boot = build_vset("boot", resample, reps=3, seed=0, output_matching=True)
    Xb, yb = boot(X, y)

    def lr(X, y):
        return np.mean(y)

    def rf(X, y):
        return np.mean(y) + 0.1

    model = Vset("model", [lr, rf], vfunc_keys=["LR", "RF"])
    report = stability_report(model(Xb, yb, n_out=1), compare="model")
    assert list(report.results.columns) == ["boot", "model", "out"]
    assert report.results["boot"].iloc[0] == "rep=0, func=resample"
    assert report.ranking["model"].iloc[0] == "RF"


def resample(X, y, random_state=None):
    idx = np.random.default_rng(random_state).integers(0, len(X), len(X))
    return X[idx], y[idx]


def test_input_checks():
    df = results({("a", "LR"): 1.0, ("a", "RF"): 0.5})
    with pytest.raises(ValueError, match="not one of the Vsets"):
        stability_report(df, compare="models")
    with pytest.raises(ValueError, match="only one choice"):
        stability_report(df, compare="boot")
    arrays = df.assign(out=[np.zeros(2), np.ones(2)])
    with pytest.raises(ValueError, match="one number per result"):
        stability_report(arrays)


def test_rendering_and_saving(tmp_path):
    report = stability_report(
        results({("a", "LR"): 0.8, ("a", "RF"): 0.9, ("b", "LR"): 0.7, ("b", "RF"): 1}),
        compare="model",
    )
    md = report.to_markdown()
    assert md.startswith("# Stability report") and "| boot |" in md
    assert str(report) == md
    assert "<h1>" in report._repr_html_() and "<html" not in report._repr_html_()

    html_path = report.save(str(tmp_path / "r.html"))
    with open(html_path, encoding="utf-8") as f:
        page = f.read()
    assert page.startswith("<!doctype html>") and "<code>model</code>" in page
    md_path = report.save(str(tmp_path / "r.md"))
    with open(md_path, encoding="utf-8") as f:
        assert f.read() == md
