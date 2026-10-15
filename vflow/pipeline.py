"""Class that stores the entire pipeline of steps in a data-science workflow
"""

import itertools

import joblib
import networkx as nx
import pandas as pd

from vflow.vset import PREV_KEY, Vset


class PCSPipeline:
    def __init__(self, steps=None, cache_dir=None):
        """
        Parameters
        ----------
        steps: list
            a list of Vset instances
        cache_dir: str, default=None
            The directory to use as data store by `joblib`. If None, won't do
            caching.
        """
        if steps is None:
            steps = []
        self.steps = steps
        # set up the cache
        self.memory = joblib.Memory(location=cache_dir)

    def run(self, *args, **kwargs):
        """Runs the pipeline: each step is called on the previous step's outputs
        (the first on `args`), and the last step's outputs are returned.

        A step that returns a tuple or list (e.g. one output dict per input)
        passes them on as separate positional arguments. `kwargs` go to every
        step. As when calling a Vset directly, a step given several inputs
        splits its output into one dict per input; for a step that combines
        its inputs into one output (e.g. a metric), pass `n_out=1`.
        """
        run_step_cached = self.memory.cache(_run_step)
        outputs = args
        for i, step in enumerate(self.steps):
            print(getattr(step, "name", f"Step {i}"))
            outputs, fitted_step = run_step_cached(step, *outputs, **kwargs)
            self.steps[i] = fitted_step
            if not isinstance(outputs, (tuple, list)):
                outputs = (outputs,)
        return outputs[0] if len(outputs) == 1 else outputs

    def __getitem__(self, i):
        """Accesses ith step of pipeline"""
        return self.steps[i]

    def __len__(self):
        return len(self.steps)

    def generate_names(self, as_pandas=True):
        """Names for every combination of vfuncs across the steps: a DataFrame
        with one column per step, or a list of tuples."""
        name_lists = []
        for step in self.steps:
            prefix = "" if as_pandas else f"{step.name}_"
            name_lists.append(
                [
                    f"{prefix}{i}_{str(mod)[:8]}"
                    for i, mod in enumerate(step.vfuncs.values())
                ]
            )
        indexes = list(itertools.product(*name_lists))
        if as_pandas:
            return pd.DataFrame(indexes, columns=[step.name for step in self.steps])
        return indexes


def build_graph(node, draw=True):
    """Builds (and optionally draws) the graph of Vsets that led to `node`,
    following the PREV_KEY lineage that Vsets record.

    Parameters
    ----------
    node: dict or Vset
        An output dict of a Vset, or a (fitted) Vset.
    draw: bool, default True
        If True, draw the graph with networkx.

    Returns
    -------
    G: nx.DiGraph
    """

    def unnest_node(node):
        """The Vset (or "init") a lineage entry refers to: entries are a
        Vset, "init", or a tuple whose first item is one of those."""
        if isinstance(node, (str, Vset)):
            return node
        if isinstance(node, tuple):
            return unnest_node(node[0])
        return None

    def add_edges(node, G):
        """Adds the edges leading to `node` to G (in place)."""
        if isinstance(node, str):
            return  # reached "init"
        if isinstance(node, dict):
            # an output dict: its PREV_KEY is (the Vset that made it, *inputs)
            nodes_prev = node[PREV_KEY]
            G.add_edge(nodes_prev[0], "End")
            for node_prev in nodes_prev[1:]:
                G.add_edge(unnest_node(node_prev), nodes_prev[0])
                add_edges(node_prev, G)
        elif isinstance(node, Vset):
            for node_prev in getattr(node, PREV_KEY, ()):
                G.add_edge(unnest_node(node_prev), node)
                add_edges(node_prev, G)
        elif isinstance(node, tuple):
            # (Vset, *the lineage of its inputs)
            func_node = unnest_node(node[0])
            add_edges(func_node, G)
            for arg_node in node[1:]:
                G.add_edge(unnest_node(arg_node), func_node)
                add_edges(arg_node, G)

    G = nx.DiGraph()
    add_edges(node, G)
    if draw:
        nx.draw(G, with_labels=True, node_color="#CCCCCC")
    return G


def _run_step(step, *args, **kwargs):
    # always call the step: returning a fitted step's vfuncs would hand Vfunc
    # objects, not outputs, to the next step (the pipeline cache avoids rework)
    outputs = step(*args, **kwargs)
    return outputs, step
