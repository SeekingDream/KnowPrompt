
import openai
import argparse
# python label_data_with_gpt.py --api_key "Your openai.api_key" --data_id 0 --prompt_id 0 --llm_id 0
from prompt_template import CodeTranslationTemplate
from utils import common_load_dataset


def getargument():
    parser = argparse.ArgumentParser(description='OpenAI API for knowPrompt')
    parser.add_argument('--api_key', type=str, required=True, help='Your openai.api_key')
    parser.add_argument('--data_id', type=int, required=True, help='data_id')
    parser.add_argument('--prompt_id', type=int, required=True, help='prompt_id')
    parser.add_argument('--llm_id', type=int, required=True, help='llm_id')
    args = parser.parse_args()
    return args

def get_instruction(prompt_id):
    method_name = f'get_instruction_{prompt_id}'
    method = getattr(template, method_name)
    result = method('some code', 'source language', 'target language')
    return result

llm = ['gpt-3.5-turbo',
       "text-davinci-003",
       "XXX"]

args = getargument()
openai.api_key = args.api_key
data_id = args.data_id
prompt_id = args.prompt_id
llm_id = args.llm_id
train_data, _ = common_load_dataset(data_id)
template = CodeTranslationTemplate()
res = []

def ask_gpt35_turbo(src_lang,tgt_lang):
    for i in range(len(train_data['x'])):
        x = train_data['x'][i]
        y = train_data['y'][i]
        instruction = template.get_instruction_1(x,src_lang,tgt_lang) #Todo
        response = openai.ChatCompletion.create(
            model=llm[llm_id],
            messages=[
                {"role": "user",
                 "content": instruction},
            ],
            temperature=0,
        )
        print(response.choices[0].message.content)
        pred = response.choices[0].message.content
        elem = {
            'x' : x,
            'y' : y,
            'gpt_pred': pred
        }
        res.append(elem)
        print(res[i])
    return res


def ask_text_davinci_003(src_lang, tgt_lang):
    for i in range(len(train_data['x'])):
        x = train_data['x'][i]
        y = train_data['y'][i]
        instruction = CodeTranslationTemplate.get_instruction_1(x, src_lang, tgt_lang)
        response = openai.Completion.create(
            model=llm[llm_id],
            prompt=instruction,
            temperature=0
        )
        pred = response["choices"][0]["text"]
        print(pred)
        elem = {
            'x': x,
            'y': y,
            'gpt_pred': pred
        }
        res.append(elem)
    return res

def ask_XXX():
    pass
