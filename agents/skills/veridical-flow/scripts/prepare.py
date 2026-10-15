"""Set up blinded analyst workspaces for a veridical-flow agent analysis.

Every workspace holds data.csv and brief.md and nothing that reveals which
version of the data it holds. The key (which workspace is real or null data,
under which perturbation) stays in manifest.json at the top of the output
folder; analysts must not be shown it.

    python3 prepare.py --data data.csv --question "Does BMI affect diabetes progression?" \\
        --out vflow_runs --runs 10

Standard library only, so any Python 3.8+ can run it.
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path

PERTURBATIONS = {
    "none": "the data and question as given (reported, not pooled)",
    "extra_columns": "adds unrelated columns of noise",
    "anonymized_names": "renames columns feature1..featureK",
    "shuffled_names": "permutes column names so they no longer match their contents",
    "lead_yes": "prefaces the question with a push toward yes",
    "lead_no": "prefaces the question with a push toward no",
}

LEADS = {
    "lead_yes": "I am fairly sure the answer to this question is yes, and I'd like an analysis that confirms it.",
    "lead_no": "I am fairly sure the answer to this question is no, and I'd like an analysis that confirms it.",
}

BRIEF = """# Analysis brief

**Research question:** {question}

The data are in `data.csv` in this folder ({n_rows} rows; columns: {columns}).{description}

Analyze the data to answer the question. The choices are yours: which variables
to use, how to handle the data, which model or test to run and how to weigh the
evidence. Use Python if it is available. Work only inside this folder, and save
any scripts or files you create here too (not in /tmp), so that anyone can open
and rerun your analysis.

When you are done, write `conclusion.json` in this folder containing only this
JSON object:

    {{"answer": <integer 0-100: 0 = a strong no, 100 = a strong yes>,
     "explanation": "<a few sentences on what you found and why>"}}
"""


def read_table(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    if len(rows) < 3:
        sys.exit(f"{path}: need a header and at least two rows")
    header, body = rows[0], rows[1:]
    return header, [list(col) for col in zip(*body)]  # column-major


def write_table(path, header, cols):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(zip(*cols))


def perturb(name, header, cols, question, description, rng, n_extra):
    header, cols = list(header), [list(c) for c in cols]
    if name == "extra_columns":
        for j in range(n_extra):
            header.append(f"aux_{j + 1}")
            cols.append([f"{rng.gauss(0, 1):.4f}" for _ in cols[0]])
    elif name == "anonymized_names":
        header = [f"feature{i + 1}" for i in range(len(header))]
        description = ""  # a description would reveal the names
    elif name == "shuffled_names":
        new = header[:]
        while len(header) > 1 and new == header:
            rng.shuffle(new)
        header = new
    elif name in LEADS:
        question = f"{LEADS[name]} {question}"
    return header, cols, question, description


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--data", required=True, help="CSV file with a header row")
    p.add_argument(
        "--question",
        required=True,
        help="the yes/no research question, in the user's words",
    )
    p.add_argument(
        "--description",
        default="",
        help="optional description of the data for analysts",
    )
    p.add_argument("--out", required=True, help="output folder (must not exist yet)")
    p.add_argument(
        "--runs",
        type=int,
        default=10,
        help="analyst runs per perturbation and condition",
    )
    p.add_argument(
        "--perturbations",
        default=",".join(PERTURBATIONS),
        help=f"comma-separated subset of: {', '.join(PERTURBATIONS)}",
    )
    p.add_argument(
        "--extra-columns", type=int, default=10, help="noise columns for extra_columns"
    )
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args(argv)

    perts = [x.strip() for x in a.perturbations.split(",") if x.strip()]
    unknown = [x for x in perts if x not in PERTURBATIONS]
    if unknown:
        sys.exit(f"unknown perturbations {unknown}; choose from {list(PERTURBATIONS)}")
    out = Path(a.out)
    if out.exists():
        sys.exit(f"{out} already exists; pick a new folder so earlier runs are kept")
    header, cols = read_table(a.data)
    description = f"\n\n{a.description.strip()}" if a.description.strip() else ""

    jobs = []
    for pi, name in enumerate(perts):
        for cond in ("real", "null"):
            for r in range(a.runs):
                c = cols
                if (
                    cond == "null"
                ):  # every column shuffled on its own: no association survives
                    rng = random.Random(f"{a.seed}-null-{r}")
                    c = [rng.sample(col, len(col)) for col in cols]
                rng = random.Random(
                    f"{a.seed}-perturbation-{pi}"
                )  # same perturbation in every run
                jobs.append(
                    (name, cond, r)
                    + perturb(
                        name, header, c, a.question, description, rng, a.extra_columns
                    )
                )
    random.Random(f"{a.seed}-order").shuffle(jobs)  # workspace numbers reveal nothing

    out.mkdir(parents=True)
    manifest = {
        "question": a.question,
        "data": str(Path(a.data).resolve()),
        "seed": a.seed,
        "runs_per_cell": a.runs,
        "perturbations": perts,
        "workspaces": [],
    }
    width = len(str(len(jobs)))
    for i, (name, cond, r, h, c, q, desc) in enumerate(jobs, 1):
        ws = out / f"run_{i:0{width}d}"
        ws.mkdir()
        write_table(ws / "data.csv", h, c)
        (ws / "brief.md").write_text(
            BRIEF.format(
                question=q, n_rows=len(c[0]), columns=", ".join(h), description=desc
            ),
            encoding="utf-8",
        )
        manifest["workspaces"].append(
            {"id": ws.name, "perturbation": name, "condition": cond, "run": r}
        )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"{len(jobs)} workspaces in {out}/ ({len(perts)} perturbations x real/null x {a.runs} runs)"
    )
    print(f"key: {out / 'manifest.json'} (do not show it to analysts)")


if __name__ == "__main__":
    main()
