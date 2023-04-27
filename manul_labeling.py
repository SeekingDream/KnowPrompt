import os
import argparse
import openai
from datasets import Dataset
from torch.utils.data import DataLoader
import torch

from tqdm import tqdm
from utils import common_load_dataset, common_split_dataset
from utils import LABEL_DATA_DIR, DATASET_LIST
from prompt_template import ManuPrmoptClassList
from src.query_gpt import query_gpt


openai.api_key = os.getenv('OPENAI_API_KEY')
if os.getenv('OPENAI_API_KEY') is None:
    print(os.environ.keys())
    exit(0)


def labeling_dataset(llm_id, prompt_func, train_data, src_lang, tgt_lang, temperature, save_dir):
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    labeled_data_all,  labeled_data_part = [], []
    data_loader = DataLoader(train_data, batch_size=1, shuffle=False)
    save_path = os.path.join(save_dir, 'tmp.tar')
    for data in tqdm(data_loader):
        try:
            x_list, y_list = data['x'], data['y']

            prompted_x_list = [prompt_func(x, src_lang, tgt_lang) for x in x_list]

            query_res = query_gpt(llm_id, prompted_x_list, temperature)

            new_res = [
                {
                    'x': x_list[i],
                    'y': query_res[i],
                    'prompted_x': prompted_x_list[i],
                    'ground_truth': y_list[i]
                }
                for i in range(len(x_list))
            ]
            part_res = [
                {
                    'x': x_list[i],
                    'y': query_res[i],
                }
                for i in range(len(x_list))
            ]
            labeled_data_all.extend(new_res)
            labeled_data_part.extend(part_res)
        except:
            pass
        torch.save([labeled_data_all, labeled_data_part], save_path)

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
        llm_id, prompt_func, train_data, src_lang, tgt_lang, temperature, save_dir)

    new_dataset_all.save_to_disk(os.path.join(save_dir, 'all_infor.csv'))
    new_dataset_part.save_to_disk(os.path.join(save_dir, 'part_infor.csv'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OpenAI API for knowPrompt')
    parser.add_argument('--dataset_id', type=int, default=0,  help='dataset id')
    parser.add_argument('--llm_id', type=int, default=0,  help='small DNN id')
    parser.add_argument('--labeling_num', type=int, default=100, help='number of training data')
    parser.add_argument('--prompt_id', type=int, default=1, help='number of training data')

    args = parser.parse_args()

    main(args)