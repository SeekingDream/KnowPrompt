import os
import argparse
import torch

from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

from utils import LOCAL_MODEL_DIR
from utils import common_load_dataset, common_split_dataset
from utils import common_tokenize_dataset
from utils import common_decoding
from src.metrics.bleu import compute_blue_scores
from train_local_model import my_data_collator


def main(args):
    dataset_id = args.dataset_id
    small_model_id = args.small_model_id
    labeling_id = args.labeling_id
    train_data_num = 10000

    _, test_data = common_load_dataset(dataset_id, labeling_id)

    test_data, _ = common_split_dataset(test_data, 1000)

    task_name = f"{dataset_id}_{small_model_id}_{labeling_id}_{train_data_num}"
    save_dir = os.path.join(LOCAL_MODEL_DIR, task_name)
    save_path = os.path.join(save_dir, 'weights')

    tokenizer = AutoTokenizer.from_pretrained(save_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(save_path)
    model = model.eval().to('cuda')

    test_data = common_tokenize_dataset(test_data, tokenizer)
    test_loader = DataLoader(test_data, batch_size=32, collate_fn=my_data_collator)
    ground_truth, predicts = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader):
            input_ids = batch['input_ids'].to('cuda')
            attention_mask = batch['attention_mask'].to('cuda')
            predict = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=256,
                num_beams=5
            )

            ground_truth.extend(common_decoding(batch['labels'].detach().to('cpu'), tokenizer))
            predicts.extend(common_decoding(predict.detach().to('cpu'), tokenizer))

    save_dir = os.path.join(save_dir, 'test')
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    s = compute_blue_scores(ground_truth, predicts, save_dir)
    print(dataset_id, small_model_id, labeling_id, s)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OpenAI API for knowPrompt')
    parser.add_argument('--dataset_id', type=int, default=1,  help='dataset id')
    parser.add_argument('--small_model_id', type=int, default=2,  help='small DNN id')
    parser.add_argument('--labeling_id', type=int, default=0, help='labeling methods')
    args = parser.parse_args()

    main(args)