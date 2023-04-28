import os
from datasets import Dataset
from src.metrics.bleu import compute_blue_scores

from utils import LABEL_DATA_DIR

train_data_num = 10000

if not os.path.isdir('tmp'):
    os.mkdir('tmp')

for llm_id in range(2):
    for prompt_id in [1, 2, 3]:
        for dataset_id in range(5):
            task_name = f"{dataset_id}_{llm_id}_{prompt_id}_{train_data_num}"
            save_dir = os.path.join(LABEL_DATA_DIR, task_name)

            datasets = Dataset.load_from_disk(os.path.join(save_dir, 'all_infor.csv'))
            prediction = datasets['y']
            ground_truth = datasets['ground_truth']
            s = compute_blue_scores(ground_truth, prediction, 'tmp')
            print(task_name, len(prediction), s)