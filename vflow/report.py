"""A one-call summary of how stable a pipeline's results are to its judgment
calls: `stability_report`.
"""

import html
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import pandas as pd

from vflow.utils import dict_to_df

OTHER = "(interactions and other)"


@dataclass(repr=False)
class StabilityReport:
    """What `stability_report` returns. Print it, show it in a notebook, or
    save it with `save`; the tables are DataFrames for further analysis."""

    headlines: list
    """Plain-language findings, one string each."""
    summary: pd.DataFrame
    """Number of results and their mean, std, min and max (per `by` group)."""
    variance: pd.DataFrame
    """Share of the variance of the results explained by each Vset on its
    own; the rest comes from interactions between Vsets."""
    levels: dict
    """For each perturbed Vset, the results summarized by its choices."""
    ranking: Optional[pd.DataFrame]
    """How often each choice of `compare` is best across the other
    perturbations (None without `compare`)."""
    results: pd.DataFrame
    """The results as a tidy table: one column per perturbed Vset."""
    wrt: str = "out"
    by: Optional[str] = None
    compare: Optional[str] = None

    def to_markdown(self) -> str:
        """The report as markdown."""
        lines = ["# Stability report", ""]
        lines += [f"- {h}" for h in self.headlines] + [""]
        for title, note, df in self._tables():
            lines += [f"## {title}", ""]
            if note:
                lines += [note, ""]
            lines += [_md_table(df), ""]
        return "\n".join(lines)

    def to_html(self, full_page: bool = True) -> str:
        """The report as HTML: a full page, or a fragment to embed."""
        parts = ['<div class="vflow-report">', "<h1>Stability report</h1>", "<ul>"]
        parts += [f"<li>{_md_code_to_html(h)}</li>" for h in self.headlines]
        parts.append("</ul>")
        for title, note, df in self._tables():
            parts.append(f"<h2>{_md_code_to_html(title)}</h2>")
            if note:
                parts.append(f"<p>{_md_code_to_html(note)}</p>")
            parts.append(
                df.to_html(
                    index=False,
                    na_rep="",
                    border=0,
                    classes="vflow-table",
                    float_format=_fmt_float,
                )
            )
        parts.append("</div>")
        body = "\n".join(parts)
        if not full_page:
            return f"<style>{_CSS}</style>\n{body}"
        return (
            '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            f"<title>Stability report</title><style>{_CSS}</style></head>"
            f"<body>\n{body}\n</body></html>\n"
        )

    def save(self, path: str) -> str:
        """Write the report to `path`: HTML for a .html or .htm path,
        markdown otherwise. Returns `path`."""
        is_html = str(path).lower().endswith((".html", ".htm"))
        text = self.to_html() if is_html else self.to_markdown()
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        return path

    def _repr_html_(self):
        return self.to_html(full_page=False)

    def __str__(self):
        return self.to_markdown()

    __repr__ = __str__

    def _tables(self):
        """(title, note, display DataFrame) for each section of the report."""
        tables = [("Summary", None, self.summary)]
        if len(self.variance):
            tables.append(
                (
                    "Where the variation comes from",
                    f"Share of the variance of `{self.wrt}` explained by each "
                    "Vset on its own.",
                    _percent(self.variance, "share"),
                )
            )
        if self.ranking is not None:
            tables.append(
                (
                    f"Ranking of `{self.compare}`",
                    f"How often each choice of `{self.compare}` is best (ties "
                    "count for every tied choice) across the combinations of "
                    "the other perturbations.",
                    _percent(self.ranking, "win_rate"),
                )
            )
        for name, df in self.levels.items():
            tables.append((f"Results by `{name}`", None, df))
        return tables


def stability_report(
    results: Union[dict, pd.DataFrame],
    compare: Optional[str] = None,
    by: Optional[str] = "auto",
    bigger_is_better: bool = True,
    wrt: str = "out",
) -> StabilityReport:
    """Summarize how stable results are to the perturbations that made them.

    Parameters
    ----------
    results: dict or pandas.DataFrame
        Output dict of a Vset whose outputs are numbers (typically a metrics
        Vset at the end of a pipeline), or a DataFrame from `dict_to_df`.
    compare: str (optional)
        Name of a Vset whose choices are being compared, e.g. the models. The
        report then says how often each choice is best across the other
        perturbations.
    by: str or None (optional), default "auto"
        Name of a Vset to report on separately rather than pool, e.g. the
        metrics (accuracy and balanced accuracy should not be averaged
        together). "auto" uses the last Vset, which produced the outputs, when
        it has more than one choice and is not `compare`; None pools all
        results.
    bigger_is_better: bool (optional), default True
        Whether larger outputs are better, for ranking `compare`.
    wrt: str (optional), default "out"
        Column holding the outputs.

    Returns
    -------
    report: StabilityReport
        Print it, show it in a notebook, or `save` it as HTML or markdown.

    Examples
    --------
    >>> metrics = metrics_set.evaluate(preds_test, y_test)
    >>> report = stability_report(metrics, compare="modeling")
    >>> report.save("stability.html")
    """
    df = _tidy(results, wrt)
    vsets = [c for c in df.columns if c != wrt]
    for name, arg in (("compare", compare), ("by", None if by == "auto" else by)):
        if arg is not None and arg not in vsets:
            raise ValueError(f"{name}={arg!r} is not one of the Vsets {vsets}")
    if by == "auto":
        # the last Vset made the outputs (e.g. the metrics): report on each
        # of its choices separately, unless it is what is being compared
        last = vsets[-1] if vsets else None
        by = last if last and last != compare and df[last].nunique() > 1 else None
    by_cols = [] if by is None else [by]
    factors = [c for c in vsets if c != by and df[c].nunique() > 1]
    if compare is not None and compare not in factors:
        raise ValueError(f"compare={compare!r} has only one choice; nothing to rank")

    groups = list(df.groupby(by, sort=False)) if by else [(None, df)]
    summary = _summary(df, by_cols, wrt)
    parts = []
    for key, g in groups:
        part = _variance_shares(g, factors, wrt)
        if by:
            part.insert(0, by, key)
        parts.append(part)
    variance = pd.concat(parts, ignore_index=True)
    levels = {f: _summary(df, by_cols + [f], wrt) for f in factors}
    ranking = None
    if compare is not None:
        others = [f for f in factors if f != compare]
        ranking = _ranking(df, compare, by_cols, others, wrt, bigger_is_better)

    headlines = []
    for key, g in groups:
        label = f"`{by}` = {key}: " if by else ""
        headlines += _headlines(
            label, g, variance, ranking, by=by, key=key, compare=compare, wrt=wrt
        )
    return StabilityReport(
        headlines=headlines,
        summary=summary,
        variance=variance,
        levels=levels,
        ranking=ranking,
        results=df,
        wrt=wrt,
        by=by,
        compare=compare,
    )


def _tidy(results, wrt):
    """One column per Vset (constant init columns dropped), readable labels,
    and numeric outputs."""
    df = dict_to_df(results) if isinstance(results, dict) else results.copy()
    if wrt not in df.columns or len(df) == 0:
        raise ValueError(f"no outputs in column {wrt!r} to report on")
    df = df.loc[:, [not str(c).startswith("init-") for c in df.columns]]
    df = df.loc[:, ~df.columns.duplicated()]
    try:
        df[wrt] = pd.to_numeric(df[wrt]).astype(float)
    except (TypeError, ValueError) as e:
        raise ValueError(
            "stability_report needs one number per result (e.g. the output of a "
            f"metrics Vset); {wrt!r} holds {type(df[wrt].iloc[0]).__name__} values"
        ) from e
    for c in df.columns:
        if c != wrt:
            df[c] = df[c].map(_label)
    return df.reset_index(drop=True)


def _label(v):
    """('rep=0', 'func=resample') -> 'rep=0, func=resample'."""
    if isinstance(v, tuple):
        return ", ".join(map(str, v))
    return str(v)


def _summary(df, cols, wrt):
    """count, mean, std, min and max of `wrt`, overall or per group of `cols`."""
    stats = ["count", "mean", "std", "min", "max"]
    if not cols:
        return df[wrt].agg(stats).to_frame().T.reset_index(drop=True)
    return df.groupby(cols, sort=False)[wrt].agg(stats).reset_index()


def _variance_shares(df, factors, wrt):
    """Between-choice sum of squares of each Vset over the total."""
    y = df[wrt]
    total = float(((y - y.mean()) ** 2).sum())
    rows = []
    for f in factors:
        means = df.groupby(f, sort=False)[wrt].agg(["mean", "size"])
        between = float((means["size"] * (means["mean"] - y.mean()) ** 2).sum())
        rows.append(
            {
                "vset": f,
                "choices": len(means),
                "share": between / total if total else 0.0,
            }
        )
    rows.sort(key=lambda r: -r["share"])
    if factors:
        explained = sum(r["share"] for r in rows)
        rows.append(
            {
                "vset": OTHER,
                "choices": np.nan,
                "share": max(0.0, 1.0 - explained) if total else 0.0,
            }
        )
    return pd.DataFrame(rows, columns=["vset", "choices", "share"])


def _ranking(df, compare, by_cols, others, wrt, bigger_is_better):
    """Rank `compare`'s choices within each combination of the others."""
    block_cols = by_cols + others
    if block_cols:
        ranks = df.groupby(block_cols, sort=False)[wrt].rank(
            ascending=not bigger_is_better, method="min"
        )
    else:
        ranks = df[wrt].rank(ascending=not bigger_is_better, method="min")
    d = df.assign(_rank=ranks, _win=(ranks == 1).astype(int))
    out = (
        d.groupby(by_cols + [compare], sort=False)
        .agg(
            mean=(wrt, "mean"),
            std=(wrt, "std"),
            mean_rank=("_rank", "mean"),
            wins=("_win", "sum"),
            blocks=("_win", "size"),
        )
        .reset_index()
    )
    out["win_rate"] = out["wins"] / out["blocks"]
    sort_cols = by_cols + ["win_rate", "mean_rank"]
    ascending = [True] * len(by_cols) + [False, True]
    if by_cols:
        # keep the by groups in their original order
        order = {k: i for i, k in enumerate(df[by_cols[0]].unique())}
        out["_order"] = out[by_cols[0]].map(order)
        sort_cols[0] = "_order"
    out = out.sort_values(sort_cols, ascending=ascending, kind="stable")
    return out.drop(columns="_order", errors="ignore").reset_index(drop=True)


def _headlines(label, g, variance, ranking, *, by, key, compare, wrt):
    """Plain-language findings for one group `g` of results (all of them
    when `by` is None)."""
    y = g[wrt]
    lines = [
        f"{label}{len(y)} results, mean {_fmt_float(y.mean())} "
        f"(from {_fmt_float(y.min())} to {_fmt_float(y.max())})."
    ]
    var = variance if by is None else variance[variance[by] == key]
    if len(var) and y.nunique() > 1:
        top = var.iloc[0] if var.iloc[0]["vset"] != OTHER else None
        other = var[var["vset"] == OTHER]["share"].sum()
        if top is not None and top["share"] >= other:
            lines.append(
                f"{label}most of the variation comes from `{top['vset']}` "
                f"({top['share']:.0%} of the variance)."
            )
        else:
            lines.append(
                f"{label}no single Vset explains most of the variation; "
                f"{other:.0%} comes from interactions between them."
            )
    elif len(var):
        lines.append(f"{label}the results do not vary across the perturbations.")
    if ranking is not None:
        r = ranking if by is None else ranking[ranking[by] == key]
        best = r.iloc[0]
        line = (
            f"{label}`{best[compare]}` is best in {int(best['wins'])} of "
            f"{int(best['blocks'])} perturbations ({best['win_rate']:.0%})"
        )
        if len(r) > 1:
            nxt = r.iloc[1]
            line += f"; next is `{nxt[compare]}` ({nxt['win_rate']:.0%})"
        lines.append(line + ".")
    return lines


def _fmt_float(v) -> str:
    """4 significant digits; empty for missing values."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    return f"{v:.4g}"


def _percent(df, col):
    """A copy of `df` with `col` shown as whole percentages."""
    out = df.copy()
    out[col] = out[col].map(lambda v: "" if pd.isna(v) else f"{v:.0%}")
    return out


def _md_table(df) -> str:
    """`df` as a markdown table, without its index."""

    def cell(v):
        if isinstance(v, (float, np.floating)):
            text = _fmt_float(float(v))
        else:
            text = str(v)
        return text.replace("|", "\\|")

    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(cell(v) for v in row) + " |")
    return "\n".join(lines)


def _md_code_to_html(text: str) -> str:
    """Escape `text` and render its `code` spans."""
    parts = html.escape(text).split("`")
    return "".join(f"<code>{p}</code>" if i % 2 else p for i, p in enumerate(parts))


_CSS = """
.vflow-report { font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  color: #1f2a33; max-width: 960px; margin: 0 auto; padding: 16px; line-height: 1.5; }
.vflow-report h1 { font-size: 1.6rem; margin: 0.4em 0; }
.vflow-report h2 { font-size: 1.15rem; margin: 1.4em 0 0.4em; }
.vflow-report code { background: #eef2f5; padding: 0 4px; border-radius: 4px; }
.vflow-report table.vflow-table { border-collapse: collapse; margin: 0.4em 0 1em;
  font-size: 0.9rem; display: block; overflow-x: auto; max-width: 100%; }
.vflow-report .vflow-table th, .vflow-report .vflow-table td { padding: 4px 10px;
  border-bottom: 1px solid #dde3e8; text-align: left; white-space: nowrap; }
.vflow-report .vflow-table th { background: #f4f7f9; }
"""
