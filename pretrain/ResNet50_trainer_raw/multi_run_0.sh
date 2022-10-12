ps -ef | grep nmp | awk -F ' ' '{print($2)}' | xargs kill -9
lsof -i:30000 | awk -F ' ' '{ if (NR>1) {print($2)}}' | xargs kill -9
rm -rf checkpoint
rm -rf tf_non_chief_save/
export DOMAIN=test
export PLATFORM=tensorflow
export JOBID=06cefa0e-2d5c-4346-b02e-9abb507023f5
export PLATFORM=tensorflow
export PYTHONPATH=/home/protago/ly/NetMind-Mixin:${PYTHONPATH}
export TIME_ESTIMATE_PROCESS=1
export MONITOR=wandb
echo $PYTHONPATH
export  ROLE=master
export INDEX=0
export TIME_ESTIMATE_PROCESS=1
export TIME_ESTIMATE_BATCH=20
export DATA_LOCATION=/data/food-101/sub_images
entry_point=train_trainer_food101_nmp.py
train_num=1300
test_num=100
train_list_path=data/list_train.txt
test_list_path=data/validation_test.txt
CUDA_VISIBLE_DEVICES="0" python $entry_point \
                          --per_device_train_batch_size=80 \
                          --learning_rate=0.05  \
                          --num_train_epochs=2  \
                          --model_name_or_path=empty


