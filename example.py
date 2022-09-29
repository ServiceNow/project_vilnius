from vilnius.evaluation import check_answer_binary, evaluate_fact_accuracy
from vilnius.fact import generate_facts
from vilnius.gpt3 import gpt3_query
from vilnius.graph import assign_names_to_nodes, generate_dag, plot_graph
from vilnius.prompt import (
    generate_binary_question_prompt,
    generate_templated_prompt_header,
)
from vilnius.question import few_shot_balanced_types, generate_all_pair_questions


G = generate_dag(n=5, p=0.4)
G = assign_names_to_nodes(G, use_real_words=False)
plot_graph(G)

facts = generate_facts(G, fact_type="v1")

prompt_header = generate_templated_prompt_header(G, facts, prompt_type="v6")

questions = generate_all_pair_questions(G, facts)

question_prompt = generate_binary_question_prompt(
    questions.iloc[0], few_shot_balanced_types(5, questions, exclude=[0])
)

prompt = prompt_header + question_prompt
print(prompt)
model_answer = gpt3_query(prompt)
print(model_answer)

print(
    "\n\nIs the answer correct?",
    check_answer_binary(questions.iloc[0]["answer"], model_answer),
)

print(
    "\n\nAre the facts correct?",
    evaluate_fact_accuracy(questions.iloc[0], model_answer),
)
