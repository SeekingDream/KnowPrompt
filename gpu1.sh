export CUDA_VISIBLE_DEVICES=6,7

python train_local_model.py \
    --dataset_id 0 \
    --small_model_id 0 \
    --labeling_id 0 \
    --training_num 10000 \
    | tee tmp_log.txt

