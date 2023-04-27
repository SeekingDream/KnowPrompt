import os
import argparse
import openai
from datasets import Dataset

from tqdm import tqdm
from utils import common_load_dataset, common_split_dataset
from utils import LABEL_DATA_DIR, DATASET_LIST
from prompt_template import ManuPrmoptClassList
from src.query_gpt import query_gpt


openai.api_key = os.getenv('OPENAI_API_KEY')
if os.getenv('OPENAI_API_KEY') is None:
    print(os.environ.keys())
    exit(0)


def labeling_dataset(llm_id, prompt_func, train_data, src_lang, tgt_lang, temperature):
    labeled_data_all,  labeled_data_part = [], []
    for data in tqdm(train_data):
        prompted_x = prompt_func(data['x'], src_lang, tgt_lang)

        query_res = query_gpt(llm_id, prompted_x, temperature)

        new_res = {
            'x': data['x'],
            'y': query_res,
            'prompted_x': prompted_x,
            'ground_truth': data['y']
        }
        part_res = {
            'x': data['x'],
            'y': query_res,
        }
        labeled_data_all.append(new_res)
        labeled_data_part.append(part_res)

    my_labeled_data_all = Dataset.from_list(labeled_data_all)
    my_labeled_data_part = Dataset.from_list(labeled_data_part)
    return my_labeled_data_all, my_labeled_data_part


def main(args):
    dataset_id = args.dataset_id
    train_data_num = args.labeling_num
    llm_id = args.llm_id
    prompt_id = args.prompt_id
    _, src_lang, tgt_lang = DATASET_LIST[dataset_id]
    prompt_class = ManuPrmoptClassList[dataset_id]
    temperature = 0

    train_data, test_data = common_load_dataset(dataset_id, label_id=0)
    train_data, _ = common_split_dataset(train_data, train_data_num)

    prompt_func = getattr(prompt_class, 'get_instruction_%d' % prompt_id)
    task_name = f"{dataset_id}_{llm_id}_{prompt_id}_{train_data_num}"
    save_dir = os.path.join(LABEL_DATA_DIR, task_name)

    new_dataset_all, new_dataset_part = labeling_dataset(
        llm_id, prompt_func, train_data, src_lang, tgt_lang, temperature)

    new_dataset_all.save_to_disk(os.path.join(save_dir, 'all_infor.csv'))
    new_dataset_part.save_to_disk(os.path.join(save_dir, 'part_infor.csv'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OpenAI API for knowPrompt')
    parser.add_argument('--dataset_id', type=int, default=0,  help='dataset id')
    parser.add_argument('--llm_id', type=int, default=0,  help='small DNN id')
    parser.add_argument('--labeling_num', type=int, default=100, help='number of training data')
    parser.add_argument('--prompt_id', type=int, default=0, help='number of training data')

    args = parser.parse_args()

    main(args)