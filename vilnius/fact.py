"""
Functions to generate facts that describe a causal graph

"""
import networkx as nx
import numpy as np


def generate_facts(
    G, fact_type="v1", include_missing_edges=False, randomize_causal_words=False
):
    """
    Generate a natural language description of a graph. This also adds the facts
    that involve each edge as edge attributes (in place).

    Parameters:
    -----------
    G: nx.DiGraph
        A causal directed acyclic graph
    include_missing_edges: bool
        Whether or not to include variables that a variable does not cause, in addition
        to stating the variables that it does cause.
    randomize_causal_worlds: bool, default=True
        Whether or not to randomize the words used to state causation in the facts.

    Returns:
    --------
    facts: list
        A list of strings representing the graph in the form of facts stated in natural language.

    """
    facts = []
    fact_by_edge = {}

    def _log_fact(fact):
        fact_id = len(facts) + 1
        facts.append((fact_id, fact))
        return fact_id

    if randomize_causal_words:
        causal_words = ["cause", "influence", "affect", "control"]
    else:
        causal_words = ["cause"]

    all_nodes = set(G.nodes())
    for node in all_nodes:
        causal_word = np.random.choice(causal_words)

        if include_missing_edges:
            raise NotImplementedError()
            # XXX: Currently disabled since requires refactoring.
        #             # No children
        #             if G.out_degree(node) == 0:
        #                 _log_fact(f"We know that {node} does not directly {causal_word} {_enum(all_nodes - {node}, 'or')}.")
        #             # Causes all other variables
        #             elif G.out_degree(node) == len(all_nodes) - 1:
        #                 _log_fact(f"We know that {node} directly {causal_word}s {_enum(G.successors(node), 'and')}.")
        #             # General case
        #             else:
        #                 children = set(G.successors(node))
        #                 _log_fact(f"We know that {node} directly {causal_word}s {_enum(children, 'and')}, but that it does not directly {causal_word} {_enum(all_nodes - {node} - children, 'or')}.")
        else:
            # Here we only state direct causal relationships and we don't mention it when a variable
            # is not the cause of another. This should be used in combination with a fact that states
            # that all causal relationships are mentioned (to avoid ambiguity).
            if G.out_degree(node) > 0:
                children = set(G.successors(node))

                if True:
                    # Fact mode: one per causal edge
                    for c in children:
                        fact_id = _log_fact(
                            f"manipulating the value of {node} causes a change in the value of {c}"
                        )  # Symbolic fact
                        #                         fact_id = _log_fact(f"cause({node}; {c})")  # Symbolic fact
                        fact_by_edge[(node, c)] = fact_id

                else:
                    # Fact mode: one per causal parent
                    #                 fact_id = _log_fact(f"{node} directly {causal_word}s {_enum(children, 'and')}")
                    #                 fact_id = _log_fact(f"Changing the value of {node} changes the value of {_enum(children, 'and')}")
                    fact_id = _log_fact(
                        f"{node} is a cause of {_enum(children, 'and')}"
                    )
                    for c in children:
                        fact_by_edge[(node, c)] = fact_id

    # Assign facts to edges. Used to generate explanations.
    nx.set_edge_attributes(G, {k: {"fact": v} for k, v in fact_by_edge.items()})

    return facts
