"""
Functions used to query GPT-3

"""
import openai
import os


try:
    openai.api_key = os.environ["OPENAI_API_KEY"]
except:
    raise RuntimeError(
        "You need to specify your OpenAI API key via the OPENAI_API_KEY environment variable."
    )


def gpt3_query(prompt, deterministic=True, model="text-davinci-002"):
    """
    Query GPT-3 for prompt completion

    """
    completion = openai.Completion.create(
        engine=model,
        prompt=prompt,
        max_tokens=250,
        temperature=0 if deterministic else None,
    )
    return completion.choices[0].text.strip()
