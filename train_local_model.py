import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES']='6,7'
import torch
import argparse
from torch.utils.data import DataLoader

from utils import common_load_dataset
from utils import common_load_dnn
from utils import common_tokenize_dataset
from utils import common_split_dataset
from utils import common_decoding
from utils import LOCAL_MODEL_DIR

from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer
from transformers import EvalPrediction

from src.metrics.bleu import compute_blue_scores



def my_data_collator(batch):
    input_ids = []
    attention_mask = []
    labels = []
    for d in batch:
        input_ids.append(torch.tensor(d['input_ids']))
        attention_mask.append(torch.tensor(d['attention_mask']))
        labels.append(torch.tensor(d['labels']))

    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "labels": torch.stack(labels),
    }


def main(args):
    dataset_id = args.dataset_id
    train_data_num = args.training_num
    small_model_id = args.small_model_id
    labeling_id = args.labeling_id

    train_data, test_data = common_load_dataset(dataset_id, labeling_id)
    train_data, _ = common_split_dataset(train_data, train_data_num)
    save_dir = os.path.join(LOCAL_MODEL_DIR, f"{dataset_id}_{small_model_id}_{labeling_id}_{train_data_num}")

    model, tokenizer = common_load_dnn(small_model_id)
    train_data = common_tokenize_dataset(train_data, tokenizer)
    test_data = common_tokenize_dataset(test_data, tokenizer)

    def compute_metrics(pred: EvalPrediction):
        ground_truth = common_decoding(pred.label_ids, tokenizer)
        predictions = common_decoding(pred.predictions, tokenizer)
        s = compute_blue_scores(ground_truth, predictions, save_dir)
        return {'blue_score': s}

    # Define the training arguments
    training_args = Seq2SeqTrainingArguments(
        max_steps=50000,
        save_total_limit=10,
        save_steps=1000,
        logging_steps=1000,
        predict_with_generate=True,
        evaluation_strategy="steps",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        generation_max_length=256,
        generation_num_beams=5,
        learning_rate=5e-5,
        warmup_steps=500,
        remove_unused_columns=False,
        output_dir=save_dir,
    )

    # Define the trainer and train the model
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=test_data,
        compute_metrics=compute_metrics,
        data_collator=my_data_collator,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(os.path.join(save_dir, 'weights'))

    with torch.no_grad():
        res = trainer.predict(test_data, max_length=256, num_beams=5)

    ground_truth = common_decoding(res.label_ids, tokenizer)
    predictions = common_decoding(res.predictions, tokenizer)
    s = compute_blue_scores(ground_truth, predictions, save_dir)
    print(small_model_id, s)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OpenAI API for knowPrompt')
    parser.add_argument('--dataset_id', type=int, default=0,  help='dataset id')
    parser.add_argument('--small_model_id', type=int, default=0,  help='small DNN id')
    parser.add_argument('--labeling_id', type=int, default=0, help='labeling methods')
    parser.add_argument('--training_num', type=int, default=100, help='number of training data')

    args = parser.parse_args()

    main(args)