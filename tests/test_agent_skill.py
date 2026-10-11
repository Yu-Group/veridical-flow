"""Tests for the scripts bundled with the veridical-flow agent skill
(agents/skills/veridical-flow/scripts), run the way an agent runs them."""

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPTS = (
    Path(__file__).resolve().parents[1]
    / "agents"
    / "skills"
    / "veridical-flow"
    / "scripts"
)
QUESTION = "Does x affect y?"


def run(script, *args):
    out = subprocess.run(
        [sys.executable, str(SCRIPTS / script), *map(str, args)],
        capture_output=True,
        text=True,
        check=False,
    )
    return out


def make_csv(path, effect=1.0, n=200, seed=0):
    rng = np.random.default_rng(seed)
    x, z = rng.standard_normal(n), rng.standard_normal(n)
    y = effect * x + 0.3 * z + rng.standard_normal(n)
    pd.DataFrame({"x": x.round(4), "z": z.round(4), "y": y.round(4)}).to_csv(
        path, index=False
    )
    return path


def prepare(tmp_path, effect=1.0, **kw):
    data = make_csv(tmp_path / "data.csv", effect)
    args = ["--data", data, "--question", QUESTION, "--out", tmp_path / "runs"]
    for k, v in kw.items():
        args += [f"--{k.replace('_', '-')}", v]
    out = run("prepare.py", *args)
    assert out.returncode == 0, out.stderr
    return tmp_path / "runs"


def answer_all(runs, analyst):
    """Play the analysts: each sees only its workspace."""
    for ws in sorted(p for p in runs.iterdir() if p.is_dir()):
        df = pd.read_csv(ws / "data.csv")
        brief = (ws / "brief.md").read_text()
        a = analyst(df, brief, ws)
        if a is not None:
            (ws / "conclusion.json").write_text(
                json.dumps({"answer": a, "explanation": "scripted"})
            )


def honest(df, brief, ws):
    """Tests the two columns the question names; with unhelpful names, the strongest pair."""
    q = brief.split("**Research question:**")[1].splitlines()[0]
    named = [c for c in df.columns if f" {c} " in f" {q.replace('?', ' ')} "]
    if len(named) >= 2:
        r = abs(df[named[0]].corr(df[named[1]]))
    else:
        r = df.corr().abs().where(~np.eye(df.shape[1], dtype=bool)).max().max()
    t = r * np.sqrt((len(df) - 2) / max(1e-12, 1 - r**2))
    return 95 if t > 3 else 10


def yes_sayer(df, brief, ws):
    return 70 + (int(ws.name.split("_")[1]) * 7) % 21


def summary(runs):
    out = run("summarize.py", runs)
    assert out.returncode == 0, out.stderr
    return json.loads((runs / "summary.json").read_text()), out.stdout


class TestPrepare:
    def test_workspaces_are_blinded(self, tmp_path):
        runs = prepare(tmp_path, runs=2)
        manifest = json.loads((runs / "manifest.json").read_text())
        ws = [w for w in manifest["workspaces"]]
        assert len(ws) == 6 * 2 * 2
        assert {w["condition"] for w in ws} == {"real", "null"}
        for w in ws:
            d = runs / w["id"]
            assert sorted(p.name for p in d.iterdir()) == ["brief.md", "data.csv"]
            text = (d / "brief.md").read_text().lower()
            assert "null" not in w["id"] and "real" not in w["id"]
            for word in ("shuffl", "null", "perturb", "manifest"):
                assert word not in text, f"{word!r} would reveal the design"
        conds = [w["condition"] for w in ws]
        assert conds != sorted(conds), "workspace order must not follow the condition"

    def test_null_shuffles_every_column(self, tmp_path):
        runs = prepare(tmp_path, runs=1, perturbations="none")
        m = json.loads((runs / "manifest.json").read_text())
        real = pd.read_csv(
            runs
            / next(w["id"] for w in m["workspaces"] if w["condition"] == "real")
            / "data.csv"
        )
        null = pd.read_csv(
            runs
            / next(w["id"] for w in m["workspaces"] if w["condition"] == "null")
            / "data.csv"
        )
        orig = pd.read_csv(tmp_path / "data.csv")
        assert real.equals(orig)
        for c in orig.columns:
            assert sorted(null[c]) == sorted(orig[c]) and not null[c].equals(orig[c])
        assert abs(null.x.corr(null.y)) < 0.2

    def test_perturbations(self, tmp_path):
        runs = prepare(tmp_path, runs=2, extra_columns=4)
        m = json.loads((runs / "manifest.json").read_text())
        get = (
            lambda p, k=0: runs
            / [
                w["id"]
                for w in m["workspaces"]
                if w["perturbation"] == p and w["condition"] == "real"
            ][k]
        )
        assert list(pd.read_csv(get("extra_columns") / "data.csv").columns) == [
            "x",
            "z",
            "y",
            "aux_1",
            "aux_2",
            "aux_3",
            "aux_4",
        ]
        assert list(pd.read_csv(get("anonymized_names") / "data.csv").columns) == [
            "feature1",
            "feature2",
            "feature3",
        ]
        shuffled = list(pd.read_csv(get("shuffled_names") / "data.csv").columns)
        assert sorted(shuffled) == ["x", "y", "z"] and shuffled != ["x", "z", "y"]
        assert shuffled == list(
            pd.read_csv(get("shuffled_names", 1) / "data.csv").columns
        ), "same in every run"
        assert (
            "fairly sure the answer to this question is yes"
            in (get("lead_yes") / "brief.md").read_text()
        )
        assert "is no" in (get("lead_no") / "brief.md").read_text()
        assert QUESTION in (get("none") / "brief.md").read_text()

    def test_refuses_existing_folder_and_bad_names(self, tmp_path):
        prepare(tmp_path, runs=1, perturbations="none")
        again = run(
            "prepare.py",
            "--data",
            tmp_path / "data.csv",
            "--question",
            QUESTION,
            "--out",
            tmp_path / "runs",
        )
        assert again.returncode != 0 and "already exists" in again.stderr
        bad = run(
            "prepare.py",
            "--data",
            tmp_path / "data.csv",
            "--question",
            QUESTION,
            "--out",
            tmp_path / "r2",
            "--perturbations",
            "nope",
        )
        assert bad.returncode != 0 and "unknown perturbations" in bad.stderr


class TestSummarize:
    def test_honest_analyst_gets_stable_yes(self, tmp_path):
        runs = prepare(tmp_path, runs=3)
        answer_all(runs, honest)
        s, printed = summary(runs)
        assert s["pooled"]["verdict"] == "stable yes", s["pooled"]
        assert s["pooled"]["enough_runs"] and s["pooled"]["missing"] == 0
        assert s["by_perturbation"]["none"]["mean_real"] == 95
        assert "Verdict: stable yes" in printed
        rows = list(csv.DictReader(open(runs / "results.csv")))
        assert len(rows) == 36 and {r["condition"] for r in rows} == {"real", "null"}

    def test_honest_analyst_on_noise(self, tmp_path):
        runs = prepare(
            tmp_path, effect=0.0, runs=3, perturbations="none,extra_columns,lead_yes"
        )
        answer_all(runs, honest)
        assert summary(runs)[0]["pooled"]["verdict"] == "no evidence for yes"

    def test_yes_sayer_is_caught(self, tmp_path):
        runs = prepare(tmp_path, runs=3)
        answer_all(runs, yes_sayer)
        p = summary(runs)[0]["pooled"]
        assert p["signal"] and not p["separation"]
        assert p["verdict"] == "yes, but no different from shuffled data"

    def test_missing_and_malformed_answers(self, tmp_path):
        runs = prepare(tmp_path, runs=6, perturbations="none")
        ws = sorted(p for p in runs.iterdir() if p.is_dir())
        (ws[0] / "conclusion.json").write_text(
            'Here you go: {"answer": 80, "explanation": "ok"}'
        )
        (ws[1] / "conclusion.json").write_text('{"answer": "high"}')
        (ws[2] / "conclusion.json").write_text('{"answer": 250}')
        s, printed = summary(runs)
        rows = {r["id"]: r for r in csv.DictReader(open(runs / "results.csv"))}
        assert (
            rows[ws[0].name]["answer"] == "80.0"
            and rows[ws[2].name]["answer"] == "100.0"
        )
        assert s["pooled"]["missing"] == 10, "one malformed plus nine never written"
        assert not s["pooled"]["enough_runs"] and "smoke test" in printed

    def test_overlap_and_signal_math(self):
        sys.path.insert(0, str(SCRIPTS))
        try:
            import summarize as sm
        finally:
            sys.path.pop(0)
        rng = np.random.default_rng(0)
        a = list(rng.normal(80, 5, 100))
        assert sm.overlap(a, a) == pytest.approx(1, abs=0.01)
        assert sm.overlap(a, list(rng.normal(20, 5, 100))) < 0.01
        assert sm.overlap([100.0] * 20, [100.0] * 20) > 0.45
        ok, p = sm.signal_check([100.0] * 20, n_boot=999)
        assert ok and p == pytest.approx(1 / 1000)
        assert sm.signal_check([50.0] * 20) == (False, 1.0)
        assert not sm.signal_check([30.0, 60.0, 45.0, 55.0, 40.0])[0]
