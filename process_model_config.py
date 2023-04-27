import json
import os

from transformers import AutoConfig
from utils import MODEL_CONFIG_DIR

model_name_list = [
    'Salesforce/codet5-small',          # XXXX
    'Salesforce/codet5-base',           # XXXX
    "mrm8488/bert2bert_shared-spanish-finetuned-summarization"   # XXXX
]

config_list = []
for model_id, model_name in enumerate(model_name_list):
    config = AutoConfig.from_pretrained(model_name)
    model_name = model_name.replace('/', '_')
    save_path = os.path.join(MODEL_CONFIG_DIR, model_name)
    config.save_pretrained(save_path)
    config_list.append(config)

