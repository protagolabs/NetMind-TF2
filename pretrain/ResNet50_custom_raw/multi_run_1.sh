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
CUDA_VISIBLE_DEVICES="1" python test_multi_worker.py  --category_num=1000 \
                          --per_device_train_batch_size=100 --weight_decay=0.0001 --label_smoothing=0.1 --train_num=1300 \
                          --test_num=100 --learning_rate=0.05 --minimum_learning_rate=0.0001 \
                          --save_steps=2  --num_train_epochs=2 --warmup_steps=5 \
                          --do_predict=True --per_device_eval_batch_size=10 --gradient_accumulation_steps=10 \
                          --adam_beta1=10 --adam_epsilon=10 --max_grad_norm=1.0 --max_steps=5 --warmup_ratio=1 \
                          --logging_steps=10 --fp16=False --model_name_or_path=empty --train_list_path=data/train_test.txt --test_list_path=data/validation_test.txt

#python test_multi_worker.py  0
