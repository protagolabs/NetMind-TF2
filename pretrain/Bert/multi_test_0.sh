lsof -i:30000 | awk -F ' ' '{ if (NR>1) {print($2)}}' | xargs kill -9
rm -rf checkpoint
rm -rf tf_non_chief_save/
export DOMAIN=test
export PLATFORM=tensorflow
export JOBID=06cefa0e-2d5c-4346-b02e-9abb507023f5
export PLATFORM=tensorflow
export PYTHONPATH=/home/protago/ly/NetMind-Images/NetmindModelEnv/NetmindMixins:${PYTHONPATH}
#export TIME_ESTIMATE_PROCESS=1
export MONITOR=wandb
echo $PYTHONPATH
export  ROLE=master
export INDEX=0
#export TIME_ESTIMATE_PROCESS=1
#export TIME_ESTIMATE_BATCH=20
entry_point=train_netmind.py
CUDA_VISIBLE_DEVICES="0" python $entry_point  --category_num=1000 \
                          --per_device_train_batch_size=100 --label_smoothing=0.1 --train_num=12811 \
                          --test_num=500  --minimum_learning_rate=0.0001 \
                          --save_steps=2 --warmup_steps=5 \
                          --do_predict=True  --gradient_accumulation_steps=10 \
                          --adam_beta1=10 --adam_epsilon=10 --max_grad_norm=1.0 --max_steps=5 --warmup_ratio=1 \
                          --logging_steps=10 --fp16=False  --train_list_path=data/train_test.txt --test_list_path=data/validation_test.txt \
                          --num_train_epochs=6 --per_device_eval_batch_size=9 --learning_rate=0.0001 --warmup_proportion=0.1 \
                          --weight_decay=1e-7 --output_dir=./bert_saved_model_ep6 --max_seq_length=512 \
                          --preprocessing_num_workers=128  --overwrite_cache=True --config_name=bert-base-uncased \
                          --tokenizer_name=bert-base-uncased --model_name_or_path=""
