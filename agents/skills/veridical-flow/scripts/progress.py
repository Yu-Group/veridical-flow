"""Show how far a veridical-flow agent analysis has got.

    python3 progress.py vflow_runs                 # one progress line
    python3 progress.py vflow_runs --pending 8     # ...and the next 8 workspaces to hand out
    python3 progress.py vflow_runs --watch 30      # redraw every 30 s until every run is in

A workspace is answered once it holds a usable conclusion.json, in progress
once an analyst has written anything else into it, and not started otherwise.
Elapsed time runs from when prepare.py wrote the workspaces; the ETA assumes
the remaining runs finish at the rate the answered ones did.

Standard library only, so any Python 3.8+ can run it.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize import (
    read_answer,
)  # noqa: E402  (same folder; one definition of "answered")

STARTING_FILES = {"data.csv", "brief.md"}


def status(root):
    """Counts and lists for every workspace in the manifest."""
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    answered, unusable, running, waiting = [], [], [], []
    for w in manifest["workspaces"]:
        ws = root / w["id"]
        if (ws / "conclusion.json").exists():
            (answered if read_answer(ws)[0] is not None else unusable).append(w["id"])
        elif any(p.name not in STARTING_FILES for p in ws.iterdir()):
            running.append(w["id"])
        else:
            waiting.append(w["id"])
    total = len(manifest["workspaces"])
    elapsed = time.time() - manifest.get("created", time.time())
    done = len(answered) + len(unusable)
    eta = None
    if 0 < done < total:
        eta = elapsed / done * (total - done)
    return {
        "total": total,
        "answered": len(answered),
        "unusable": len(unusable),
        "in_progress": len(running),
        "not_started": len(waiting),
        "elapsed_s": round(elapsed),
        "eta_s": None if eta is None else round(eta),
        "complete": done == total,
        "not_started_ids": waiting,
        "in_progress_ids": running,
    }


def duration(seconds):
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s" if m else f"{s}s"


def glyphs(stream=None):
    """Block characters for the bar where the output can show them, plain ASCII
    where it cannot (a Windows console or pipe defaults to cp1252, which has no
    block characters, and printing one there would crash)."""
    encoding = getattr(stream or sys.stdout, "encoding", None) or "ascii"
    try:
        "█░·".encode(encoding)
        return "█", "░", "·"
    except (UnicodeEncodeError, LookupError):
        return "#", "-", "|"


def bar(st, width=24):
    full, empty, sep = glyphs()
    done = st["answered"] + st["unusable"]
    filled = round(width * done / st["total"]) if st["total"] else width
    parts = [
        f"[{full * filled}{empty * (width - filled)}]",
        f"{st['answered']}/{st['total']} answered ({100 * done // max(st['total'], 1)}%)",
    ]
    if st["unusable"]:
        parts.append(f"{st['unusable']} unusable")
    if st["in_progress"]:
        parts.append(f"{st['in_progress']} in progress")
    if st["not_started"]:
        parts.append(f"{st['not_started']} not started")
    parts.append(f"elapsed {duration(st['elapsed_s'])}")
    if st["complete"]:
        parts.append("done: run summarize.py")
    elif st["eta_s"] is not None:
        parts.append(f"ETA ~{duration(st['eta_s'])}")
    return f" {sep} ".join([parts[0] + " " + parts[1]] + parts[2:])


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("runs", help="the folder prepare.py made")
    p.add_argument(
        "--pending",
        type=int,
        default=0,
        metavar="N",
        help="also list up to N not-started workspaces (absolute paths) to hand out next",
    )
    p.add_argument(
        "--watch",
        type=float,
        default=0,
        metavar="SECONDS",
        help="redraw every SECONDS until every workspace is answered",
    )
    p.add_argument("--json", action="store_true", help="print the counts as JSON")
    a = p.parse_args(argv)

    root = Path(a.runs).resolve()
    if not (root / "manifest.json").exists():
        sys.exit(f"no manifest.json in {root}; run prepare.py first")

    while True:
        st = status(root)
        if a.json:
            print(json.dumps({k: v for k, v in st.items() if not k.endswith("_ids")}))
        elif a.watch and sys.stdout.isatty():
            print("\r" + bar(st) + " " * 4, end="", flush=True)
        else:
            print(bar(st), flush=True)
        if not a.watch or st["complete"]:
            break
        time.sleep(a.watch)
    if a.watch and sys.stdout.isatty() and not a.json:
        print()

    if a.pending and st["not_started_ids"]:
        print(f"\nnext {min(a.pending, len(st['not_started_ids']))} to hand out:")
        for i in st["not_started_ids"][: a.pending]:
            print(f"  {root / i}")


if __name__ == "__main__":
    main()
