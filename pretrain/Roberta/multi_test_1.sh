#lsof -i:30000 | awk -F ' ' '{ if (NR>1) {print($2)}}' | xargs kill -9
rm -rf checkpoint
rm -rf tf_non_chief_save/
export DOMAIN=test
export PLATFORM=tensorflow
export JOBID=06cefa0e-2d5c-4346-b02e-9abb507023f5
export PLATFORM=tensorflow
export PYTHONPATH=/home/protago/ly/NetMind-Mixin:${PYTHONPATH}
export MONITOR=wandb
echo $PYTHONPATH
export INDEX=1
#export TIME_ESTIMATE_PROCESS=1
#export TIME_ESTIMATE_BATCH=20
entry_point=train_netmind.py
export DATA_LOCATION=./data_roberta
CUDA_VISIBLE_DEVICES="1" python $entry_point  \
                          --adam_beta1=10 \
                          --adam_epsilon=10 \
                          --num_train_epochs=5 \
                          --learning_rate=0.0001 \
                          --warmup_proportion=0.1 \
                          --weight_decay=0.01 \
                          --config_name=roberta-base \
                          --tokenizer_name=roberta-base \
                          --per_device_train_batch_size=16
