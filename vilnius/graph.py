"""
Graph generation functions

"""
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pycorpora


def assign_names_to_nodes(G, use_real_words=True):
    """
    Assigns names to the variables in the causal graph, which are later
    used to state the fact in natural language and formulate questions.

    """
    if use_real_words:
        words = pycorpora.words.nouns["nouns"]
        labels = np.random.choice(words, len(G.nodes()), replace=False)
    else:
        labels = [f"X{i}" for i in range(len(G.nodes()))]

    return nx.relabel_nodes(G, dict(zip(G.nodes(), labels)))


def generate_dag(n, p=0.2):
    """
    Generate a random Erdos-Reyni DAG

    """
    A = np.tril((np.random.rand(n, n) < p).astype(float), -1)
    P = np.random.permutation(n)
    A = A[P][:, P]
    G = nx.DiGraph(A)
    assert nx.is_directed_acyclic_graph(G), "Graph is not acyclic."
    return G


def load_graph(filename):
    """
    Load a graph from a file in edgelist format

    """
    return nx.DiGraph(nx.read_edgelist(filename, create_using=nx.DiGraph))


def plot_graph(G):
    """
    Plot a graph in current matplotlib figure

    """
    plt.clf()
    nx.draw(G, with_labels=True, node_color="red", node_size=1000)
    plt.show()


def save_graph(G, filename):
    """
    Save a graph into the NetworkX edgelist format

    """
    nx.write_edgelist(G, filename)


def save_graph_to_dot(G, filename):
    """
    Save graph to a dot file in markup language. Useful for sharing and visualization
    via https://dreampuf.github.io/GraphvizOnline.

    """
    nx.drawing.nx_pydot.write_dot(G, filename)
