
# File Structure
    ## label_data_with_gpt.py   
        --data_id  
        --prompt_id
        --llm_id 

    ## LabeledDaraset/
        LabeledDaraset/{LLM_id}-{task_id}-{prompt_id}/train
        LabeledDaraset/{LLM_id}-{task_id}-{prompt_id}/test

    ## preprocess_dataset.py    preprocess spider dataset
    
    ## prompt_template.py       manul defined prompt

    ## train_local_model.py
        --small_dnn_id
        --labeled_data_id
    
    ## utils.py              common used function


# Step
    1. get the ground truth dataset for each task.  
        spider dataset needs specifical preprocess

    2. query gpt and save to ```LabeledDaraset```

    3. train local model using the dataset from ```LabeledDaraset```.
    

