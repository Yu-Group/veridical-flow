"""Class that stores the entire pipeline of steps in a data-science workflow
"""
import itertools

import joblib
import networkx as nx
import pandas as pd

from vflow.vset import PREV_KEY


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
        """Runs the pipeline
        """
        run_step_cached = self.memory.cache(_run_step)
        for i, step in enumerate(self.steps):
            try:
                step_name = step.name
            except AttributeError:
                step_name = f'Step {i}'
            print(step_name)
            _, fitted_step = run_step_cached(step, *args, **kwargs)
            self.steps[i] = fitted_step

    def __getitem__(self, i):
        """Accesses ith step of pipeline
        """
        return self.steps[i]

    def __len__(self):
        return len(self.steps)

    def generate_names(self, as_pandas=True):
        name_lists = []
        if as_pandas:
            for step in self.steps:
                name_lists.append([f'{i}_{str(mod)[:8]}'
                                   for i, mod in enumerate(step)])
            indexes = list(itertools.product(*name_lists))
            return pd.DataFrame(indexes, columns=[step.name for step in self.steps])
        for step in self.steps:
            name_lists.append([f'{step.name}_{i}_{str(mod)[:8]}'
                               for i, mod in enumerate(step)])
        return list(itertools.product(*name_lists))


def build_graph(node, draw=True):
    """Helper function that just calls build_graph_recur with an empty graph

    Parameters
    ----------
    node: dict or Vset

    Returns
    -------
    G: nx.Digraph()
    """

    def unnest_node(node):
        """Unnest a node, if necessary (i.e., when node is a tuple)

        Parameters
        ----------
        node: str, dict, Vset, or tuple

        Returns
        -------
        unnested_node: str, Vset, or None
        """
        node_type = type(node)
        if node_type is str or 'Vset' in str(node_type):
            return node
        if node_type is tuple:
            return unnest_node(node[0])
        return None

    def build_graph_recur(node, G):
        """Builds a graph up using __prev__ and PREV_KEY pointers

        Parameters
        ----------
        node: str, dict, Vset, or tuple
        G: nx.Digraph()

        Returns
        -------
        G: nx.Digraph()
        """
        # base case: reached starting node
        if isinstance(node, str):
            return G

        # initial case: starting at dict
        if isinstance(node, dict):
            s_node = 'End'
            nodes_prev = node[PREV_KEY]
            G.add_edge(nodes_prev[0], s_node)
            for node_prev in nodes_prev[1:]:
                G.add_edge(unnest_node(node_prev), nodes_prev[0])
                G = build_graph_recur(node_prev, G)
            return G

        # main case: at a moduleset
        if 'Vset' in str(type(node)):
            if hasattr(node, PREV_KEY):
                nodes_prev = getattr(node, PREV_KEY)
                for node_prev in nodes_prev:
                    G.add_edge(unnest_node(node_prev), node)
                    G = build_graph_recur(node_prev, G)
            return G

        # nested prev key case
        if isinstance(node, tuple):
            func_node = unnest_node(node[0])
            G = build_graph_recur(func_node, G)
            for arg_node in node[1:]:
                G.add_edge(unnest_node(arg_node), func_node)
                G = build_graph_recur(arg_node, G)
            return G

        return G

    G = nx.DiGraph()
    G = build_graph_recur(node, G)
    if draw:
        nx.draw(G, with_labels=True, node_color='#CCCCCC')
    return G


def _run_step(step, *args, **kwargs):
    if step._fitted:
        return step.modules, step
    outputs = step(*args, **kwargs)
    return outputs, step
