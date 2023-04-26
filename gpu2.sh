export CUDA_VISIBLE_DEVICES=2,3

DATA_ID=0
SMALL_DNN_ID=1
LABELING_ID=0
TRAINING_NUM=10000


python train_local_model.py \
--dataset_id $DATA_ID \
--small_model_id $SMALL_DNN_ID \
--labeling_id $LABELING_ID \
--training_num $TRAINING_NUM \
| tee log/$DATA_ID"_"$SMALL_DNN_ID"_"$LABELING_ID"_"$TRAINING_NUM.txt


DATA_ID=1

python train_local_model.py \
--dataset_id $DATA_ID \
--small_model_id $SMALL_DNN_ID \
--labeling_id $LABELING_ID \
--training_num $TRAINING_NUM \
| tee log/$DATA_ID"_"$SMALL_DNN_ID"_"$LABELING_ID"_"$TRAINING_NUM.txt




