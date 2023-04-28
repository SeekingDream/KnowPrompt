import os.path
import random
import numpy as np

from datasets import load_dataset
from datasets import Dataset
from transformers import AutoTokenizer
from transformers import AutoConfig
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoModelForCausalLM

from src.data_tuils import preprocess_translation
from src.data_tuils import preprocess_summarization


DATASET_LIST = [
    ("CM/codexglue_codetrans", 'java', 'c#'),    # C2JAVA
    ("CM/codexglue_codetrans", 'c#', 'java'),    # C2JAVA


    ("CM/codexglue_code2text_go", 'go', 'nl'),
    ("CM/codexglue_code2text_java", 'java', 'nl'),
    ("CM/codexglue_code2text_python", 'python', 'nl'),
]

SMALL_DNN_LIST = [
    ('Salesforce_codet5-small', 'Salesforce/codet5-small', AutoModelForSeq2SeqLM),          # XXXX
    ('Salesforce_codet5-base', 'Salesforce/codet5-base', AutoModelForSeq2SeqLM),          # XXXX

    ("mrm8488_bert2bert-6", "mrm8488/bert2bert_shared-spanish-finetuned-summarization", AutoModelForSeq2SeqLM),
    ("mrm8488_bert2bert-12", "mrm8488/bert2bert_shared-spanish-finetuned-summarization", AutoModelForSeq2SeqLM),
    # ("google_bert2bert_L-6", "google/bert2bert_L-24_wmt_de_en", AutoModelForSeq2SeqLM),
    # ("google_bert2bert_L-12", "google/bert2bert_L-24_wmt_de_en", AutoModelForSeq2SeqLM)
    #TODO
]


LOCAL_MODEL_DIR = 'model_weights'
MODEL_CONFIG_DIR = 'model_config'
LABEL_DATA_DIR = 'labeled_data'

if not os.path.isdir(LOCAL_MODEL_DIR):
    os.mkdir(LOCAL_MODEL_DIR)
if not os.path.isdir(MODEL_CONFIG_DIR):
    os.mkdir(MODEL_CONFIG_DIR)
if not os.path.isdir(LABEL_DATA_DIR):
    os.mkdir(LABEL_DATA_DIR)



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


def common_manu_prompted_dataset(dataset_id, prompt_id, llm_id):
    train_data_num = 10000
    task_name = f"{dataset_id}_{llm_id}_{prompt_id}_{train_data_num}"
    save_dir = os.path.join(LABEL_DATA_DIR, task_name)
    dataset = Dataset.load_from_disk(os.path.join(save_dir, 'part_infor.csv'))
    return dataset, None


def common_load_dataset(dataset_id, label_id):
    if label_id == 0:
        return common_load_groundtruth_dataset(dataset_id)
    elif label_id in [1, 2, 3]:
        return common_manu_prompted_dataset(dataset_id, label_id, llm_id=0)
    elif label_id in [4, 5, 6]:
        return common_manu_prompted_dataset(dataset_id, label_id, llm_id=1)
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
    np.random.seed(1997)
    if data_num is None:
        return dataset, None
    all_indices = list(range(len(dataset)))
    np.random.shuffle(all_indices)
    select_index = all_indices[:data_num]
    rest_index = all_indices[data_num:]

    return dataset.select(select_index), dataset.select(rest_index)


def common_load_dnn(model_id):
    local_model_name, tokenizer_name, model_class = SMALL_DNN_LIST[model_id]
    config_path = os.path.join(MODEL_CONFIG_DIR, local_model_name)

    config = AutoConfig.from_pretrained(config_path)
    model = model_class.from_config(config)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)   # 'Salesforce/codet5-base'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def common_decoding(predictions, tokenizer):
    predictions[predictions == -100] = tokenizer.pad_token_id
    return tokenizer.batch_decode(predictions, skip_special_tokens=True)


def common_compute_model_size(model):
    sum_p = 0
    for p in model.parameters():
        sum_p += p.numel()
    return sum_p / 10 ** 6


if __name__ == '__main__':
    for model_id in range(4):
        model, tokenizer = common_load_dnn(model_id)
        print(model_id, 'successful', common_compute_model_size(model))


