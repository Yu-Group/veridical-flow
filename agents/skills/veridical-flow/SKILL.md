---
name: veridical-flow
description: veridical-flow for agents - PCS stability analysis for a yes/no conclusion an AI agent draws from data. Use this whenever an agent (you, a subagent, or another AI analyst) has answered or is about to answer a data question such as "does X affect Y?", "is there an effect of the treatment?" or "is A associated with B?" and the answer will be reported or acted on; whenever the user asks how robust, reproducible, stable or trustworthy an AI-generated analysis is, or whether its finding could come from noise; and whenever the user asks to stress-test, double-check or vet an agent's data analysis, even if they don't say "PCS" or "veridical". It re-runs the analysis with independent analyst agents on the real data, on shuffled data and under perturbations, then says whether the conclusion holds up.
---

# veridical-flow for agents

One analysis is one draw. Ask a capable agent the same data question twice and
it can make different reasonable choices and reach different answers. Ask it
about data whose columns have been shuffled into noise and it can still say
"yes". A conclusion deserves trust when it holds across reasonable choices and
looks different from what the same analyst says about noise. That is the PCS
idea behind veridical-flow, applied to the agent itself:

- **Predictability:** answers on the real data must stand apart from answers on
  shuffled (null) data, where the true answer is "no" by construction.
- **Computability:** every run is a saved folder anyone can open and rerun.
- **Stability:** the answer should survive perturbations that should not
  matter: unrelated extra columns, anonymized or misleading column names, and a
  question that leans toward one answer.

The workflow: write blinded workspaces, have independent analysts fill them,
then score the answers. Two bundled scripts (standard-library Python) handle
the parts that must be exact; you do the orchestration and the reporting.

## 1. Frame the question

You need a CSV file and a yes/no question about it. Keep the user's wording:
sharpening or softening the question changes what is being tested. If the
question is not yes/no ("what drives churn?"), ask the user to pick the yes/no
claim they care about ("does contract length affect churn?").

## 2. Agree on the size of the run

Each workspace is one full agent analysis, so cost and time add up quickly.
The count is `runs x perturbations x 2` (real and shuffled data):

| Use | `--runs` | Perturbations | Analyst runs |
|---|---|---|---|
| Smoke test that the setup works | 1 | `none` | 2 |
| Quick look | 3 | all 6 | 36 |
| A result to report | 10 or more | all 6 | 120 or more |

Tell the user the number of analyst runs and which model the analysts will use,
and get their go-ahead before launching more than about 20. If they want it
cheap, a smaller model for the analysts is fine; say which you used.

## 3. Write the workspaces

```bash
python3 <skill-dir>/scripts/prepare.py --data data.csv \
    --question "Does BMI affect diabetes progression?" \
    --description "442 patients; target is disease progression after one year" \
    --out vflow_runs --runs 3
```

This writes `vflow_runs/run_001/ ...`, each holding `data.csv` and `brief.md`,
in random order. Half the workspaces hold the real data and half a copy with
every column shuffled independently. Each perturbation (`none`,
`extra_columns`, `anonymized_names`, `shuffled_names`, `lead_yes`, `lead_no`;
choose with `--perturbations`) is applied to both. Which workspace is which is
recorded only in `vflow_runs/manifest.json`.

## 4. Run one independent analyst per workspace

Give each workspace to a fresh analyst agent, with this task and nothing else:

> Read `brief.md` in `<absolute path to the workspace>` and follow it. Work
> only inside that folder. Finish by writing `conclusion.json` there.

Independence and blinding are what make the comparison meaningful, so:

- **Use a new agent for every workspace.** With a subagent tool (Agent/Task),
  launch them in parallel batches. Without one, run a fresh CLI session in each
  folder, e.g. `claude -p "Read brief.md and follow it." --allowedTools "Bash(python3:*)" Read Write`
  with the workspace as the working directory. An analyst that has seen other
  workspaces anchors on them, and the spread you are trying to measure
  disappears.
- **Do not answer the briefs yourself** in this conversation, for the same
  reason, and because you know the key.
- **Say nothing about the design.** No hint that data may be shuffled, no
  mention of the manifest, the perturbations or your own view. An analyst that
  knows it might be looking at noise behaves differently from one doing a real
  analysis, and the real analysis is what is being measured.
- Analysts run code on this machine. If the data or setup is untrusted, run
  them in a sandbox or container.

A workspace with no usable `conclusion.json` is fine: it is counted as missing.
Do not rerun it selectively or fill in an answer.

## 5. Score

```bash
python3 <skill-dir>/scripts/summarize.py vflow_runs
```

It writes `results.csv` and `summary.json` and prints one row per perturbation
plus a pooled row (all perturbations except `none`). Two checks on the pooled
answers decide the verdict:

- **Signal:** the mean answer on real data is above 50 (one-sided bootstrap,
  p < 0.05).
- **Separation:** real and shuffled-data answers overlap by less than 0.2
  (0 = the two sets of answers never overlap, 1 = identical).

| | Separation passes | Separation fails |
|---|---|---|
| **Signal passes** | stable yes | yes, but no different from shuffled data |
| **Signal fails** | some signal, not enough for yes | no evidence for yes |

## 6. Report

Lead with the verdict in plain words and what it means for the user's question,
then give:

- mean answer on real vs shuffled data, the signal p-value and the overlap
- the number of analyst runs, the agent and model, and how many gave no answer
- any perturbation whose row disagrees with the pooled verdict, and what that
  suggests: a shift under `lead_yes`/`lead_no` means the analyst follows the
  asker; under `anonymized_names`/`shuffled_names`, it leans on column names
  rather than the data; under `extra_columns`, it is distracted by noise
- the path to the runs folder, so anyone can open a single analysis

Be plain about what the result supports. "Yes, but no different from shuffled
data" means the analyst's "yes" carries no information about this data, even if
the answer happens to be right. If `summarize.py` says there were too few
answers per condition, call the result a smoke test. Do not drop runs that
disagree or round a failed check into a pass.
