import networkx as nx
import numpy as np
import pandas as pd
import sys

from itertools import permutations
from time import sleep, time

sys.path.append("./")  # Run from top level dir of project

from vilnius.evaluation import check_answer_binary, evaluate_fact_accuracy
from vilnius.fact import generate_facts
from vilnius.gpt3 import gpt3_query
from vilnius.graph import assign_names_to_nodes, generate_dag, plot_graph
from vilnius.prompt import (
    generate_binary_question_prompt,
    generate_templated_prompt_header,
)
from vilnius.question import few_shot_balanced_types, generate_all_pair_questions


# Generate a graph with a fixed structure
# G = nx.from_numpy_array(np.array([[0, 1, 1, 0],
#                                   [0, 0, 1, 0],
#                                   [0, 0, 0, 1],
#                                   [0, 0, 0, 0]]), create_using=nx.DiGraph)


np.random.seed(3)
G = generate_dag(n=6, p=0.4)

# A good randomly generated graph
# np.random.seed(3)
# G = generate_dag(n=10, p=0.4)
# TODO: don't iterate over all permutations as I currently do since there are too many

f = plot_graph(G)
f.savefig(f"prompt_selection_{time()}.png")

print("Types of questions and abundance:")
print(generate_all_pair_questions(G, facts=generate_facts(G)).type.value_counts())
print()
sleep(5)

trial_accuracies = []
missed_answers = []
question_answers = []

for real_words in [False]:
    for shot in [-1]:  # Convention: -1 means zero shot with no CoT prompting
        for prompt_type in ["v6", "v7", "v8"]:
            for fact_type in ["v1", "v2", "v3"]:
                for trial, permutation in enumerate(
                    set(permutations(range(len(G.nodes()))))
                ):
                    G = assign_names_to_nodes(G, use_real_words=real_words)

                    permutation = np.random.permutation(len(G.nodes()))

                    # Shuffle the nodes labels (used to assign various Xi labels to the same graph structure)
                    G = nx.relabel_nodes(
                        G, dict(zip(G.nodes(), np.array(G.nodes())[permutation]))
                    )

                    facts = generate_facts(G, fact_type=fact_type)
                    prompt_header = generate_templated_prompt_header(
                        G, facts, prompt_type=prompt_type
                    )
                    print(prompt_header)

                    questions = generate_all_pair_questions(G, facts=facts)

                    correct = 0
                    for qidx, q in questions.iterrows():
                        print("Question", qidx + 1, "of", questions.shape[0])
                        few_shot_examples = few_shot_balanced_types(
                            shot if shot > 0 else 0, questions, exclude=[qidx]
                        )

                        question_prompt = generate_binary_question_prompt(
                            q, few_shot_examples
                        )

                        # Query the model
                        if shot == 0:
                            print("---> ZERO SHOT CoT")
                            # If zero-shot, we use zero-shot chain of thought prompting, which requires a special procedure
                            model_cot, model_answer = gpt3_query_zscot(
                                prompt_header + question_prompt, example="yes/no"
                            )
                            print(
                                question_prompt,
                                model_cot,
                                "\nFinal answer: ",
                                model_answer,
                            )
                            model_answer = (
                                model_cot + " So the answer is: " + model_answer
                            )
                        else:
                            print(f"---> {shot} SHOT")
                            model_answer = gpt3_query(
                                prompt_header + question_prompt, deterministic=True
                            )
                            print(
                                "Type:", q["type"], "\n", question_prompt, model_answer
                            )

                        is_correct = check_answer_binary(q["answer"], model_answer)
                        correct += int(is_correct)
                        # if not is_correct:
                        #     missed_answers.append(
                        #         (question_prompt, model_answer, q["answer"])
                        #     )
                        #     print("Missed:", missed_answers[-1])

                        tmp = dict(q)
                        tmp.update(
                            dict(
                                real_words=real_words,
                                shot=shot,
                                prompt_type=prompt_type,
                                fact_type=fact_type,
                                trial=trial,
                                permutation=permutation,
                                model_answer=model_answer,
                                is_correct=is_correct,
                            )
                        )
                        question_answers.append(tmp)
                        print("\n" * 2)
                    accuracy = correct / len(questions)
                    print(accuracy)
                    trial_accuracies.append((real_words, shot, accuracy))
                    pd.DataFrame(question_answers).to_csv(
                        f"prompt_selection_results_{time()}.csv"
                    )
