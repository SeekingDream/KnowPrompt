import openai


def ask_gpt35_turbo(question, temperature=0):
    messages = [
        {
            "role": "user",
            "content": f"{question}"
        }
    ]
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=messages,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=temperature
    )
    return response['choices'][0]['message']['content']


def ask_text_advinci_003(question, temperature=0):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=question,
        max_tokens=1024,
        temperature=temperature
    )
    pred = response["choices"][0]["text"]
    return pred


def query_gpt(llm_id, question, temperature):
    if llm_id == 0:
        return ask_gpt35_turbo(question, temperature)
    elif llm_id == 1:
        return ask_text_advinci_003(question, temperature)
    else:
        raise NotImplementedError