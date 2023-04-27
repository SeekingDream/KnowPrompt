import openai


def ask_gpt35_turbo(question_list, temperature=0):
    messages = [
        {
            "role": "user",
            "content": f"{question}"
        }
        for question in question_list
    ]
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=messages,
        max_tokens=1024,
        stop=None,
        temperature=temperature
    )
    pred = [d['message']['content'] for d in response['choices']]
    assert len(pred) == len(question_list)
    return pred


def ask_text_advinci_003(question_list, temperature=0):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=question_list,
        max_tokens=1024,
        temperature=temperature
    )
    pred = [d["text"] for d in response["choices"]]
    assert len(pred) == len(question_list)
    return pred


def query_gpt(llm_id, question, temperature):
    if llm_id == 0:
        return ask_gpt35_turbo(question, temperature)
    elif llm_id == 1:
        return ask_text_advinci_003(question, temperature)
    else:
        raise NotImplementedError