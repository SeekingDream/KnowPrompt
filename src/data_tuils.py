

def preprocess_translation(dataset, lang_name):
    def preprocess_function(examples):
        if lang_name == 'java':
            src_lang, tgt_lang = 'java', 'cs'
        elif lang_name == 'c#':
            src_lang, tgt_lang = 'cs', 'java'
        else:
            raise NotImplementedError
        src_code = examples[src_lang]
        tgt_code = examples[tgt_lang]
        examples['x'] = src_code
        examples['y'] = tgt_code
        return examples

    dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names,
        load_from_cache_file=False,
        desc="Running preprocess on dataset",
    )
    return dataset


def preprocess_summarization(dataset):
    def preprocess_function(examples):

        codes = [' '.join(d).replace('\n', ' ') for d in examples['code_tokens']]
        codes = [' '.join(code.strip().split()) for code in codes]

        docs = [' '.join(d).replace('\n', '') for d in examples['docstring_tokens']]
        docs = [' '.join(doc.strip().split()) for doc in docs]

        examples['x'] = codes
        examples['y'] = docs
        return examples

    dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on dataset",
    )
    return dataset


def preprocess_text2seq(dataset):
    def preprocess_function(examples):
        raise NotImplementedError

    dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on dataset",
    )
    return dataset
