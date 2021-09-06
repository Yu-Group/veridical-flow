'''Class that stores the entire pipeline of steps in a data-science workflow
'''
import itertools
import networkx as nx
import pandas as pd
from vflow.module_set import PREV_KEY
import matplotlib.pyplot as plt
import joblib

class PCSPipeline:
    def __init__(self, steps: list = [], cache_dir=None):
        '''Helper function that just calls build_graph_recur with an empty graph
        Params
        ------
        steps: list
            a list of ModuleSet instances
        cache_dir: str, default=None
            The directory to use as data store by joblib. If None, won't do
            caching.

        Returns
        -------
        G: nx.Digraph()
        '''
        self.steps = steps
        # set up the cache
        self.memory = joblib.Memory(location=cache_dir)

    def run(self, *args, **kwargs):
        '''Runs the pipeline
        '''
        run_step_cached = self.memory.cache(_run_step)
        for i, step in enumerate(self.steps):
            try:
                step_name = step.name
            except:
                step_name = f'Step {i}'
            print(step_name)
            outputs, fitted_step = run_step_cached(step, *args, **kwargs)
            self.steps[i] = fitted_step

    def __getitem__(self, i):
        '''Accesses ith step of pipeline
        '''
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
        else:
            for step in self.steps:
                name_lists.append([f'{step.name}_{i}_{str(mod)[:8]}'
                                   for i, mod in enumerate(step)])
            return list(itertools.product(*name_lists))
        

def build_graph(node, draw=True):
    '''Helper function that just calls build_graph_recur with an empty graph
    Params
    ------
    node: dict or ModuleSet

    Returns
    -------
    G: nx.Digraph()
    '''
    
    def build_graph_recur(node, G):
        '''Builds a graph up using __prev__ and PREV_KEY pointers
        Params
        ------
        node: dict or ModuleSet
        G: nx.Digraph()

        Returns
        -------
        G: nx.Digraph()
        '''
        # base case: reached starting node
        if type(node) is str:
            return G

        # initial case: starting at dict
        elif type(node) is dict:
            s_node = 'End'
            nodes_prev = node[PREV_KEY]
            if type(nodes_prev) is not list:
                nodes_prev = [nodes_prev]
            for node_prev in nodes_prev:
                G.add_edge(node_prev, s_node)
                G = build_graph_recur(node_prev, G)
            return G

        # main case: at a moduleset
        elif 'ModuleSet' in str(type(node)):
            # print(node)
            nodes_prev = node.__prev__
            if type(nodes_prev) is not list:
                nodes_prev = [nodes_prev]
            for node_prev in nodes_prev:
                G.add_edge(node_prev, node)
                G = build_graph_recur(node_prev, G)
            return G    

    G = nx.DiGraph()
    G = build_graph_recur(node, G)
    if draw:
        nx.draw(G, with_labels=True, node_color='#CCCCCC')
        plt.tight_layout()
    return G

def _run_step(step, *args, **kwargs):
    if step._fitted:
        return step.modules, step
    outputs = step(*args, **kwargs)
    return outputs, step
