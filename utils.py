import os.path
import random
import numpy as np

from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoConfig
from transformers import AutoModelForSeq2SeqLM

from src.data_tuils import preprocess_translation
from src.data_tuils import preprocess_summarization


DATASET_LIST = [
    ("CM/codexglue_codetrans", 'java', 'cs'),    # C2JAVA
    ("CM/codexglue_codetrans", 'cs', 'java'),    # C2JAVA


    ("CM/codexglue_code2text_go", 'go', 'nl'),
    ("CM/codexglue_code2text_java", 'java', 'nl'),
    ("CM/codexglue_code2text_python", 'python', 'nl'),


]

LOCAL_MODEL_DIR = 'model_weights'

if not os.path.isdir(LOCAL_MODEL_DIR):
    os.mkdir(LOCAL_MODEL_DIR)


SMALL_DNN_LIST = [
    'Salesforce/codet5-small',                 # XXXX
    "el-profesor/bert_small_seq2seq",          # XXXX
    'sumedh/lstm-seq2seq',
    'Salesforce/codet5-base',                 # XXXX

    # "kleinay/qanom-seq2seq-model-baseline",    # XXXX
    #TODO
]


def common_load_groundtruth_dataset(dataset_id):
    dataset_url, src_lang_name, tgt_lang_name = DATASET_LIST[dataset_id]
    if dataset_url.startswith('CM/codexglue_codetrans'):
        train_dataset = load_dataset(dataset_url, split='train', cache_dir='./Dataset')
        test_dataset = load_dataset(dataset_url, split='test', cache_dir='./Dataset')
        train_dataset = preprocess_translation(train_dataset, src_lang_name)
        test_dataset = preprocess_translation(test_dataset, src_lang_name)
    elif dataset_url.startswith('CM/codexglue_code2text'):
        train_dataset = load_dataset(dataset_url, src_lang_name, split='train', cache_dir='./Dataset')
        test_dataset = load_dataset(dataset_url, src_lang_name, split='test', cache_dir='./Dataset')
        train_dataset = preprocess_summarization(train_dataset)
        test_dataset = preprocess_summarization(test_dataset)
    else:
        raise NotImplementedError
    return train_dataset, test_dataset


def common_load_dataset(dataset_id, label_id):
    if label_id == 0:
        return common_load_groundtruth_dataset(dataset_id)
    elif label_id == 1:
        raise NotImplementedError
    else:
        raise NotImplementedError


def common_tokenize_dataset(dataset, tokenizer, max_length=256):
    def _preprocess(examples):
        tokenized_x = tokenizer(
            examples['x'],
            padding='max_length',
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
            return_tensors="pt"
        )
        examples['input_ids'] = tokenized_x.data['input_ids']
        examples['attention_mask'] = tokenized_x.data['attention_mask']

        tokenized_y = tokenizer(
            examples['y'],
            padding='max_length',
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
            return_tensors="pt"
        )
        labels = tokenized_y.data['input_ids']
        labels[labels == tokenizer.pad_token_id] = -100
        examples['labels'] = labels
        return examples

    return dataset.map(
            _preprocess,
            batched=True,
            load_from_cache_file=False,
            remove_columns=dataset.column_names,
            desc="Running tokenizer on dataset",
        )


def common_split_dataset(dataset, data_num):
    if data_num is None:
        return dataset, None
    all_indices = list(range(len(dataset)))
    np.random.shuffle(all_indices)
    select_index = all_indices[:data_num]
    rest_index = all_indices[data_num:]

    return dataset.select(select_index), dataset.select(rest_index)


def common_load_dnn(model_id):
    model_name = SMALL_DNN_LIST[model_id]
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_config(config)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer


def common_decoding(predictions, tokenizer):
    predictions[predictions == -100] = tokenizer.pad_token_id
    return tokenizer.batch_decode(predictions, skip_special_tokens=True)


def common_compute_model_size(model):
    sum_p = 0
    for p in model.parameters():
        sum_p += p.numel()
    return sum_p


if __name__ == '__main__':
    train_dataset, _ = common_load_dataset(2)
    llm_id
    instruction_id
    10000
    for data in train_dataset:
        print(data['x'], )
