"""Summarize a veridical-flow agent analysis once analysts have written
conclusion.json in their workspaces.

    python3 summarize.py vflow_runs

Writes results.csv (one row per workspace) and summary.json, and prints a
table. Two checks, on answers pooled over the perturbations (not "none"):

  signal       the mean answer on the real data is above 50
               (one-sided bootstrap, 10,000 resamples, p < alpha)
  separation   answers on the real data overlap answers on the shuffled
               (null) data by less than 0.2 (overlap of Gaussian kernel
               density estimates on 0-100; 0 = disjoint, 1 = identical)

Standard library only, so any Python 3.8+ can run it.
"""

import argparse
import csv
import json
import math
import random
import re
import sys
from pathlib import Path

VERDICTS = {
    (True, True): "stable yes",
    (True, False): "yes, but no different from shuffled data",
    (False, True): "some signal, not enough for yes",
    (False, False): "no evidence for yes",
}
MIN_RUNS = 10  # answers per condition below which the result is only a smoke test


def read_answer(ws):
    """The 0-100 answer in ws/conclusion.json (None if missing or unusable)."""
    f = Path(ws) / "conclusion.json"
    if not f.exists():
        return None, ""
    text = f.read_text()
    try:
        obj = json.loads(text)
    except ValueError:  # tolerate stray text around the object
        m = re.search(r"\{.*\}", text, re.S)
        try:
            obj = json.loads(m.group(0)) if m else {}
        except ValueError:
            obj = {}
    value = obj.get("answer", obj.get("response")) if isinstance(obj, dict) else None
    try:
        return min(100.0, max(0.0, float(value))), str(obj.get("explanation", ""))
    except (TypeError, ValueError):
        return None, ""


def signal_check(answers, threshold=50.0, alpha=0.05, n_boot=10_000, seed=42):
    """(passes, p): p = (#{bootstrap means <= threshold} + 1) / (n_boot + 1)."""
    if not answers:
        return False, 1.0
    rng, n = random.Random(seed), len(answers)
    below = sum(sum(rng.choices(answers, k=n)) / n <= threshold for _ in range(n_boot))
    p = (below + 1) / (n_boot + 1)
    return p < alpha, p


def _kde(xs, grid):
    n = len(xs)
    mean = sum(xs) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in xs) / (n - 1)) if n > 1 else 0.0
    h = sd * n ** (-1 / 5) if sd > 0 else 1.0  # Scott's rule; 1 point wide if all equal
    c = 1 / (n * h * math.sqrt(2 * math.pi))
    return [c * sum(math.exp(-0.5 * ((g - x) / h) ** 2) for x in xs) for g in grid]


def overlap(a, b, n_grid=1000):
    """Area under the smaller of the two densities on [0, 100]."""
    if not a or not b:
        return float("nan")
    grid = [100 * i / (n_grid - 1) for i in range(n_grid)]
    m = [min(x, y) for x, y in zip(_kde(a, grid), _kde(b, grid))]
    step = grid[1] - grid[0]
    return sum((m[i] + m[i + 1]) / 2 * step for i in range(n_grid - 1))


def checks(real, null, alpha, max_overlap):
    sig, p = signal_check(real, alpha=alpha)
    ovl = overlap(real, null)
    sep = bool(ovl < max_overlap)
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    return {
        "n_real": len(real),
        "n_null": len(null),
        "mean_real": mean(real),
        "mean_null": mean(null),
        "p_signal": p,
        "signal": sig,
        "overlap": ovl,
        "separation": sep,
        "verdict": VERDICTS[(sig, sep)],
        "enough_runs": min(len(real), len(null)) >= MIN_RUNS,
    }


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("runs", help="the folder prepare.py made")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--max-overlap", type=float, default=0.2)
    a = p.parse_args(argv)

    root = Path(a.runs)
    try:
        manifest = json.loads((root / "manifest.json").read_text())
    except FileNotFoundError:
        sys.exit(f"no manifest.json in {root}; run prepare.py first")
    rows = []
    for w in manifest["workspaces"]:
        ans, why = read_answer(root / w["id"])
        rows.append({**w, "answer": ans, "explanation": why})
    with open(root / "results.csv", "w", newline="") as f:
        cw = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "perturbation",
                "condition",
                "run",
                "answer",
                "explanation",
            ],
        )
        cw.writeheader()
        cw.writerows(rows)

    def answers(pert, cond):
        return [
            r["answer"]
            for r in rows
            if r["answer"] is not None
            and r["condition"] == cond
            and (
                pert is None
                and r["perturbation"] != "none"
                or r["perturbation"] == pert
            )
        ]

    per = {
        pt: checks(answers(pt, "real"), answers(pt, "null"), a.alpha, a.max_overlap)
        for pt in manifest["perturbations"]
    }
    pooled_on = (
        None if any(pt != "none" for pt in manifest["perturbations"]) else "none"
    )
    pooled = checks(
        answers(pooled_on, "real"), answers(pooled_on, "null"), a.alpha, a.max_overlap
    )
    pooled["missing"] = sum(r["answer"] is None for r in rows)
    pooled["total"] = len(rows)
    summary = {
        "question": manifest["question"],
        "pooled": pooled,
        "by_perturbation": per,
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"Question: {manifest['question']}")
    print(f"{'perturbation':<18}{'real':>6}{'null':>6}{'p':>8}{'overlap':>9}  verdict")
    for name, c in list(per.items()) + [("pooled", pooled)]:
        print(
            f"{name:<18}{c['mean_real']:6.1f}{c['mean_null']:6.1f}{c['p_signal']:8.4f}"
            f"{c['overlap']:9.2f}  {c['verdict']}"
        )
    print(
        f"\nVerdict: {pooled['verdict']}  ({pooled['missing']} of {pooled['total']} runs gave no answer)"
    )
    if not pooled["enough_runs"]:
        print(
            f"Fewer than {MIN_RUNS} answers per condition: treat this as a smoke test, not a verdict."
        )


if __name__ == "__main__":
    main()
