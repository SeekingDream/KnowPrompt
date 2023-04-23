import os
import time

import openai
import argparse

# python label_data_with_gpt.py --api_key "Your openai.api_key" --data_id 0 --prompt_id 0 --llm_id 0


parser = argparse.ArgumentParser(description='OpenAI API for knowPrompt')
parser.add_argument('--api_key', type=str, required=True, help='Your openai.api_key')
parser.add_argument('--data_id', type=int, required=True, help='data_id')
parser.add_argument('--prompt_id', type=int, required=True, help='prompt_id')
parser.add_argument('--llm_id', type=int, required=True, help='llm_id')
args = parser.parse_args()


openai.api_key = args.api_key
data_id = args.data_id
prompt_id = args.prompt_id
llm_id = args.llm_id
data = ["Your file path of Code Translation",
        "Your file path of Text-to-SQL Generation",
        "Your file path of Code Summarization"]
prompt = ["translate the following c# to java , return it in one line and don not change the syntax:\n",
          "translate the following java to c# , return it in one line and don not change the syntax:\n",
          "Todo",
          "..."]
llm = ['gpt-3.5-turbo',
       "text-davinci-003",
       "code-davinci-002"]


def read_file_lines(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return [line.strip() for line in lines]

lines = read_file_lines(data[data_id])

if(llm_id == 0):
    for i in lines:
        response=openai.ChatCompletion.create(
          model=llm[llm_id],
          messages=[
                {"role": "user",
                 "content": prompt[prompt_id]+i},
            ],
          temperature=0,
        )
        print(response.choices[0].message.content)
        dirname = f'./LabeledDataset/{llm_id}-{data_id}-{prompt_id}/train/'
        file_name = "output.txt"
        os.makedirs(dirname, exist_ok=True)
        with open(dirname+file_name, "a") as f:
            f.write(response.choices[0].message.content + '\n')
        time.sleep(0.3)

elif(llm_id == 1):
    for i in lines:
        response = openai.Completion.create(
            model=llm[llm_id],
            prompt=prompt[prompt_id],
            temperature=0
        )
        print(response.choices[0].message.content)
        dirname = f'./LabeledDataset/{llm_id}-{data_id}-{prompt_id}/train/'
        file_name = "output.txt"
        os.makedirs(dirname, exist_ok=True)
        with open(dirname + file_name, "a") as f:
            f.write(response.choices[0].message.content + '\n')
        time.sleep(0.3)
elif(llm_id == 2):
    pass
