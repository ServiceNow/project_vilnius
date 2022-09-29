"""
Functions used to evaluate model answers

"""
import numpy as np
import re


def standardize(text):
    """
    Standardize strings of text to allow comparison.

    """
    return text.lower()


def check_answer_binary(true_answer, answer):
    """
    Currently simply checks that the answer starts with the right yes/no answer.
    TODO: This fails if the model doesn't use these words or that structure.

    """
    true_answer = standardize(true_answer)
    answer = standardize(answer)
    return re.search(f"\\b{true_answer}\\b", answer) is not None


def evaluate_fact_accuracy(question, answer):
    """
    Check if list of facts is ok.
    Notes: sometimes the model only returns numbers, e.g., because of 2,3.
           Need to account for this in evaluation.

    """
    # Use regular expression to find all facts mentioned with their number.
    # The regex will match any enumeration of facts: fact(s) x[, y, z]
    matches = re.findall(
        r"(?:\bfacts?)\s\d+(?:,\s*\d+)*", answer.lower(), re.IGNORECASE
    )

    # Extract fact numbers
    answer_facts = set(
        [
            int(y.strip())
            for x in matches
            for y in x.lower().replace("facts", "").replace("fact", "").split(",")
        ]
    )

    if len(question["supporting_facts"]) == 0:
        # TODO: deal with the fact where there are no supporting facts. In this case, any fact is a FP.
        raise NotImplementError()  # I lost my implementation of this. See stackoverflow link in docs.
    else:
        # Relevant facts are stored in question["supporting_facts"] as a list of sets.
        # Each set corresponds to a different valid explanation.
        metrics_by_set = []

        for facts in question["supporting_facts"]:
            facts = [int(f) for f in facts]

            # TODO: calculate metrics: accuracy, precision, recall, f1score
            metrics = dict(
                tp=len([f for f in answer_facts if f in facts]),
                fp=len([f for f in answer_facts if f not in facts]),
                fn=len([f for f in facts if f not in answer_facts]),
            )

            metrics["precision"] = metrics["tp"] / (metrics["tp"] + metrics["fp"])
            metrics["recall"] = metrics["tp"] / (metrics["tp"] + metrics["fn"])
            metrics["f1"] = (
                2
                * (metrics["precision"] * metrics["recall"])
                / (metrics["precision"] + metrics["recall"])
            )

            metrics_by_set.append(metrics)

    return metrics_by_set[np.argmin([m["f1"] for m in metrics_by_set])]
