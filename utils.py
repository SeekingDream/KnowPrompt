import random
import numpy as np

from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoConfig
from transformers import AutoModelForSeq2SeqLM

from src.data_tuils import preprocess_translation
from src.data_tuils import preprocess_summarization


DATASET_LIST = [
    ("code_x_glue_cc_code_to_code_trans", 'java'),    # C2JAVA
    ("code_x_glue_cc_code_to_code_trans", 'cs'),    # C2JAVA


    ("code_x_glue_ct_code_to_text", 'go'),
    ("code_x_glue_ct_code_to_text", 'java'),
    ("code_x_glue_ct_code_to_text", 'python'),

    #TODO
]


SMALL_DNN_LIST = [
    'Salesforce/codet5-small',
    "kleinay/qanom-seq2seq-model-baseline",
    "el-profesor/bert_small_seq2seq",
    'sumedh/lstm-seq2seq',
    't5-small',


    #TODO
]


def common_load_dataset(dataset_id):
    dataset_url, lang_name = DATASET_LIST[dataset_id]
    if dataset_url == 'code_x_glue_cc_code_to_code_trans':
        train_dataset = load_dataset(dataset_url, split='train')
        test_dataset = load_dataset(dataset_url, split='test')
        train_dataset = preprocess_translation(train_dataset, lang_name)
        test_dataset = preprocess_translation(test_dataset, lang_name)
    elif dataset_url == "code_x_glue_ct_code_to_text":
        train_dataset = load_dataset(dataset_url, lang_name, split='train')
        test_dataset = load_dataset(dataset_url, lang_name, split='test')
        train_dataset = preprocess_summarization(train_dataset)
        test_dataset = preprocess_summarization(test_dataset)
    else:
        raise NotImplementedError
    return train_dataset, test_dataset


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
    common_load_dataset(2)
