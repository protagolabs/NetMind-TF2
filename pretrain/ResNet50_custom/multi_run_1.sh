lsof -i:30001 | awk -F ' ' '{ if (NR>1) {print($2)}}' | xargs kill -9
rm -rf checkpoint
export DOMAIN=test
export PLATFORM=tensorflow
export JOBID=06cefa0e-2d5c-4346-b02e-9abb507023f5
export PLATFORM=tensorflow
export PYTHONPATH=/home/protago/ly/NetMind-Mixin:${PYTHONPATH}
export MONITOR=wandb
echo $PYTHONPATH
export INDEX=1
export TIME_ESTIMATE_PROCESS=1
export TIME_ESTIMATE_BATCH=20
export DATA_LOCATION=/data/resnet50/extract
CUDA_VISIBLE_DEVICES="1" python train_netmind.py  \
                          --per_device_train_batch_size=100  --label_smoothing=0.1 --train_num=1300 \
                          --test_num=100 --learning_rate=0.05 --minimum_learning_rate=0.0001 \
                          --save_steps=2  --num_train_epochs=2  \
                          --model_name_or_path=empty --train_list_path=data/train_test.txt \
                          --test_list_path=data/validation_test.txt
