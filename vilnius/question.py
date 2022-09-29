"""
Functions used to generate questions

"""
import networkx as nx
import numpy as np
import pandas as pd

from .utils import _capfirst, _enum


def log_question(t, q, a, questions, qid=""):
    """
    type, question, answer

    """
    if (q, a) not in questions:
        questions[(q, a)] = (t, qid)

    return questions


def generate_all_pair_questions(G, facts):
    """
    Generates questions about the causal relationships that exist between any pair of variables
    and tags them by difficulty (i.e., length of the causal chain).
    Not very computationally efficient, but easy to understand.

    """
    facts = dict(facts)
    questions = []
    for s in G.nodes():
        for t in set(G.nodes()) - {s}:
            symbol = f"cause({s}; {t})"

            # Find all paths of causation
            all_paths = list(nx.all_simple_paths(G, s, t))

            if len(all_paths) > 0:
                # Case: a causal path exists from {s} to {t}
                # ---------------------------------------------

                # There exists a causal path, so the answer is yes.
                answer = "yes"

                # Shortest path first
                all_paths = np.array(all_paths, dtype=object)[
                    np.argsort([len(x) for x in all_paths])
                ]

                # Gather the combination of relevant facts along each causal path
                valid_fact_sets = [
                    [str(G[path[i]][path[i + 1]]["fact"]) for i in range(len(path) - 1)]
                    for path in all_paths
                ]

                # Determine the explanation used for few-shot learning and the kind of question using
                # the shortest causal path.
                shortest_path = all_paths[0]
                # TODO: pretty printing function
                explanation = (
                    _enum(
                        [
                            f"we know from Fact {G[shortest_path[i]][shortest_path[i + 1]]['fact']} that {facts[G[shortest_path[i]][shortest_path[i + 1]]['fact']]}"
                            for i in range(len(shortest_path) - 1)
                        ],
                        final="and",
                    )
                    + f", so acting on {s} does cause a change in {t}"
                )

                explanation = (
                    _enum(
                        [
                            f"we know from Fact {G[shortest_path[i]][shortest_path[i + 1]]['fact']} "
                            + f"that {facts[G[shortest_path[i]][shortest_path[i + 1]]['fact']]}"
                            for i in range(len(shortest_path) - 1)
                        ],
                        final="and",
                    )
                    + f", so {s} is a cause of {t} and {t} is an effect of {s}. Based on our "
                    + f"definition of causation, we know that manipulating the value of {s} "
                    + f"will cause a change in the value of {t}."
                )
                explanation = _capfirst(explanation)

                kind = f"chain_{len(shortest_path) - 1}"

            else:

                # Find all anti-causal paths
                all_paths = list(nx.all_simple_paths(G, t, s))

                if len(all_paths) > 0:
                    # Case: an anti-causal path exists from {t} to {s}
                    # --------------------------------------------------

                    # There exists an anti-causal path, so the answer is no.
                    answer = "no"

                    # Shortest path first
                    all_paths = np.array(all_paths, dtype=object)[
                        np.argsort([len(x) for x in all_paths])
                    ]

                    # Gather the combination of relevant facts along each causal path
                    valid_fact_sets = [
                        [
                            str(G[path[i]][path[i + 1]]["fact"])
                            for i in range(len(path) - 1)
                        ]
                        for path in all_paths
                    ]

                    # Determine the explanation used for few-shot learning and the kind of question using
                    # the shortest causal path.
                    shortest_path = all_paths[0]
                    explanation = (
                        _enum(
                            [
                                f"we know from Fact {G[shortest_path[i]][shortest_path[i + 1]]['fact']} "
                                + f"that {facts[G[shortest_path[i]][shortest_path[i + 1]]['fact']]}"
                                for i in range(len(shortest_path) - 1)
                            ],
                            final="and",
                        )
                        + f", so {t} is a cause of {s} and {s} is an effect of {t}. Based on our "
                        + f"definition of causation, whe know that manipulating the value of {s} "
                        + f"cannot cause a change in the value of {t} since causation is asymmetric."
                    )
                    explanation = _capfirst(explanation)
                    kind = f"chain_{len(shortest_path) - 1}_anti"

                else:
                    # Case: no undirected path exists between {s} and {t}
                    answer = "no"
                    # Based on the question formulation, the answer could be "maybe" or "no".
                    # As long as we ask: do the facts support that s causes t, the answer is no.
                    explanation = "There is no evidence of a causal relationship between these variables."
                    valid_fact_sets = []
                    kind = "chain_none"

            # To allow for multiple formulations of the same question
            queries = [
                # f"Does acting on {s} change {t}?"
                f"Based on these facts, can we say that manipulating the value of {s} will cause a change in the value of {t}?"
            ]

            questions += [
                dict(
                    symbol=symbol,
                    query=q,
                    answer=answer,
                    supporting_facts=valid_fact_sets,
                    explanation=explanation,
                    type=kind,
                )
                for q in queries
            ]

    return pd.DataFrame(questions)


def few_shot_example_sample(n, questions, exclude=[], seed=None):
    """
    Sample example questions for few-shot prompting

    """
    questions = questions.drop(exclude)
    return questions.sample(n, replace=False, random_state=np.random.RandomState(seed))


def few_shot_balanced_types(n, questions, exclude=[], seed=None):
    """
    Sample example questions for few-shot prompting, but assign equal probability to each type of question.

    """
    questions = questions.drop(exclude)
    probas = dict(1 / questions.type.value_counts() / len(questions.type.unique()))
    probas = np.array([probas[q.type] for _, q in questions.iterrows()])
    return questions.sample(
        n, replace=False, weights=probas, random_state=np.random.RandomState(seed)
    )


# XXX: This code is a good starting point to generate questions based on conditional independences
#      but it would require a refactoring now that the structure of the final dataframe has changed.
# def generate_dseparation_questions(G):
#     questions = {}

#     for a in G.nodes():
#         for b in G.nodes() - {a}:

#             # Unconditional
#             questions = log_question(f"dsep", f"Are the values of {a} and {b} independent?",
#                                      "yes" if nx.d_separated(G, {a}, {b}, {}) else "no", questions)

#             # Conditional (singletons)
#             for c in G.nodes() - {a, b}:
#                 questions = log_question(f"dsep", f"Are the values of {a} and {b} independent if we know the value of {c}?",
#                                          "yes" if nx.d_separated(G, {a}, {b}, {c}) else "no", questions)

#     return [dict(type=t, query=q, answer=a) for (q, a), t in questions.items()]


# XXX: This function has been replaced by the generate_all_pair_questions, which is less efficient but
#      more readable + flexible.
# def generate_chain_questions(G):
#     questions = {}
#     all_chains = [chain for s in G.nodes()
#                         for t in nx.descendants(G, s)
#                         for chain in list(nx.all_simple_paths(G, source=s, target=t))]
#     if len(all_chains) == 0:
#         return []  # Disregard that graph

#     max_chain_len = max(len(x) for x in all_chains)

#     chain_by_len = {i: [x for x in all_chains if len(x) == i] for i in range(2, max_chain_len + 1)}

#     # Disconnected node questions
#     for s in G.nodes():
#         if G.out_degree(s) == 0 and G.in_degree(s) == 0:
#             for t in set(G.nodes()) - {s}:
#                 qid = str(uuid.uuid4())
#                 questions = log_question("chain1_uncond", f"Does acting on {s} change {t}?", "no", questions, qid=qid)
#                 questions = log_question("chain1_uncond", f"Does acting on {t} change {s}?", "no", questions, qid=qid)

#     # Causal chain questions (forward and reverse)
#     # Note: there may be several paths between two variables. In this case, we consider the type of
#     #       the question to be the shortest length between the two variables.
#     for l in range(2, max_chain_len + 1):
#         for chain in chain_by_len[l]:
#             qid = str(uuid.uuid4())
#             questions = log_question(f"chain{l}_uncond", f"Does acting on {chain[0]} change {chain[-1]}?", "yes", questions, qid=qid)
#             questions = log_question(f"chain{l}_uncond", f"Does acting on {chain[-1]} change {chain[0]}?", "no", questions, qid=qid)

#     # Causal chain questions (intervention on mediator)
#     for l in range(3, max_chain_len + 1):
#         for chain in chain_by_len[l]:
#             for mediator in chain[1:-1]:
#                 # Check if there exists alternative paths.
#                 G_ = G.copy()
#                 G_.remove_node(mediator)

#                 if nx.has_path(G_, chain[0], chain[-1]):
#                     answer = "yes"  # There exists another causal path
#                 else:
#                     answer = "no"  # All causal paths are blocked

#                 questions = log_question(f"chain{l}_intervmed", f"Does acting on {chain[0]} change {chain[-1]} if we force {mediator} to remain constant?", answer, questions)

#     return pd.DataFrame([dict(type=t, query=q, answer=a, qid=qid) for (q, a), (t, qid) in questions.items()])


# def generate_graph_integrity_questions(G):
#     questions = generate_chain_questions(G)
#     return questions.loc[[q.type.endswith("uncond") for _, q in questions.iterrows()]]


# def generate_all_questions(G):
#     return pd.concat((generate_chain_questions(G),
#                       generate_dseparation_questions(G)), ignore_index=True)
