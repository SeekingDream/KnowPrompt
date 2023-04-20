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
    pass
