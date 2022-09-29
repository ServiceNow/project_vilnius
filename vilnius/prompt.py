"""
Functions used to generate prompts

"""
from .utils import _capfirst


def generate_templated_prompt_header(G, facts, prompt_type="v1"):
    """
    Generates the portion of the prompt that specifies the graph and variables in natural language.
    The generated prompt is templated, meaning that it is structured as a list of clearly defined facts.

    Parameters:
    -----------
    causal_sufficiency: bool, default=True
        Whether or not the prompt specifies that there are no other relevant variables than those
        included in the graph description. This aims to rule out paths of associations between
        variables that go through unobserved variables from the reasoning.

    Returns:
    --------
    prompt: str
        A prompt header which can be used to specify known causal relationships before asking questions.

    """
    #     prompt = f"Let us reason about {len(G.nodes())} variables: " + ", ".join(G.nodes()) + ".\n\n"
    #     prompt = f"You are given facts about the direct causal relationships that exist between {len(G.nodes())} " + \
    #              f"variables: {', '.join(G.nodes())}. No other direct causal relationships exist beyond those " + \
    #              f"mentioned in the facts. A variable is a cause of another if changing its value causes the value " +\
    #              f"the other to change. Causal relationships can be chained to form indirect causal relationships. " +\
    #              f"Answer the following questions with yes or no and justify your answer by stating the number of " +\
    #              f"the facts that influence your decision.\n\n"

    # Good one
    if prompt_type == "v1":
        prompt = (
            f"You are given facts about the direct causal relationships that exist between {len(G.nodes())} "
            + f"variables: {', '.join(G.nodes())}. No other direct causal relationships exist. Note that "
            + f"causation may propagate via a chain of causal relationships. Answer the following questions "
            + f"with yes or no and justify your answer by referring to relevant facts using their number.\n\n"
        )
    elif prompt_type == "v2":
        prompt = (
            f"You are given facts about the direct causal relationships that exist between {len(G.nodes())} "
            + f"variables: {', '.join(G.nodes())}. No other direct causal relationships exist. "
            + f"Answer the following questions "
            + f"with yes or no and justify your answer by referring to relevant facts using their number.\n\n"
        )
    elif prompt_type == "v3":
        prompt = (
            f"Context: You are given facts about the direct causal relationships that exist between {len(G.nodes())} "
            + f"variables: {', '.join(G.nodes())}. No other direct causal relationships exist. Note that "
            + f"causation may propagate via a chain of causal relationships.\n\n"
        )
    elif prompt_type == "v4":
        prompt = (
            f"Context: A variable X is said to be the cause of another variable Y if acting to change the value of X leads to a change in the value of Y. The facts below specify known causal relationships between {len(G.nodes())} "
            + f"variables: {', '.join(G.nodes())}.\n\n"
        )

    # Not stating direct relationships now:
    elif prompt_type == "v5":
        prompt = (
            f"Context: You are given facts about the direct causal relationships that exist between {len(G.nodes())} "
            + f"variables: {', '.join(G.nodes())}. If a variable X is a cause of a variable Y, then changing the value "
            + f"of X causes the value of Y to change, but changing the value of Y does not cause the value of X to "
            + f"change. Note that causation may propagate via chains of causal relationships, but that it cannot form cycles.\n\n"
        )
    elif prompt_type == "v6":
        prompt = (
            f"Definition: If manipulating the value of some quantity X causes the value of another quantity Y to change, we say that X is a cause of Y and that Y is an effect of X. Importantly, we assume that, if X is a cause of Y, then Y cannot be a cause of X (asymmetry of causation). Note that causation may propagate over chains of causal relationships.\n\n"
            + f"Context: You are given facts about the causal relationships that are known to exist between "
            + f"{len(G.nodes())} quantities: {', '.join(G.nodes())}.\n\n"
        )
    else:
        raise ValueError("Invalid prompt type!")

    #     prompt += "Fact 0: All direct causal relationships are mentioned in the following facts.\n"
    if prompt_type in ["v3", "v4", "v5"]:
        prompt += (
            "Facts:\n" + ".\n".join([f"F{fid}: {f}" for fid, f in facts]) + ".\n\n"
        )
    else:
        prompt += (
            ".\n".join([f"Fact {fid}: {_capfirst(f)}" for fid, f in facts]) + ".\n\n"
        )

    return prompt


def generate_binary_question_prompt(
    question, example_questions=None, ask_for_facts=True
):
    def _template_question(query, answer):
        """
        Standard format for questions and answers

        """
        out = "Question: " + query
        #         if ask_for_facts:
        #             out += " Answer with yes/no and state the number of the facts that affect your decision."
        #         else:
        #             out += " Answer with yes or no."
        out += f"\nAnswer (yes/no, facts): {answer}"
        return out

    prompt = (
        "Instructions: Answer the following questions with yes/no "
        + "and explain why using a list of facts.\n\n"
    )

    # Include example questions if any are provided
    if example_questions is not None:
        for _, q in example_questions.iterrows():
            prompt += (
                _template_question(
                    q["query"],
                    f"{q['explanation']} Hence, the answer is {q['answer']}.",
                )
                + "\n\n"
            )

    # The question for which we want an answer from the model
    prompt += _template_question(question["query"], "")

    return prompt


def generate_potential_cause_question_prompt(
    facts, question, example_questions=None, variables=None, causal_sufficiency=True
):
    """
    Ask a question of the form: what are the potential causes of

    """
    raise NotImplementedError()
