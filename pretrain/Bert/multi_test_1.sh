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
export DATA_LOCATION=./data_bert
CUDA_VISIBLE_DEVICES="1" python $entry_point  \
                          --num_train_epochs=1 \
                          --learning_rate=0.0001 \
         		   --save_steps=20 \
                          --warmup_proportion=0.1 \
                          --weight_decay=1e-7 \
                          --max_seq_length=512 \
                          --config_name=bert-base-uncased \
                          --tokenizer_name=bert-base-uncased \
                          --model_name_or_path="" \
                          --per_device_train_batch_size=16
