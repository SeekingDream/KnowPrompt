
from src.labeldata_gpt import getargument, ask_gpt35_turbo, ask_text_davinci_003
import json

# python label_data_with_gpt.py --api_key "Your openai.api_key" --data_id 0 --prompt_id 0 --llm_id 0
args = getargument()
res = []
if(args.llm_id == 0):
    res = ask_gpt35_turbo('Java','Cs')

elif(args.llm_id == 1):
    res = ask_text_davinci_003('Java','Cs')

elif(args.llm_id == 2):
    pass

dirname  =  f'./LabeledDataset/ { args.llm_id } - { args.data_id } - { args.prompt_id } /train/'

# 将数组写入到JSON文件中
with open(dirname + "my_array.json", "w") as f:
    json.dump(res, f)

# 从JSON文件中读取数组
# with open(dirname + "my_array.json", "r") as f:
#     new_arr = json.load(f)
