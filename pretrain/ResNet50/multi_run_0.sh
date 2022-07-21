lsof -i:30000 | awk -F ' ' '{ if (NR>1) {print($2)}}' | xargs kill -9
rm -rf checkpoint
export PLATFORM=tensorflow
export JOBID=f31bf13b-c0aa-4ff5-8882-805e4e94c6f5
export USE_DDP=1
export RANK=0
export LOCAL_RANK=0
export PLATFORM=tensorflow
export PYTHONPATH=/home/protago/ly/NetMind-Images/NetmindModelEnv/NetmindMixins:${PYTHONPATH}
#export TIME_ESTIMATE_PROCESS=1
export MONITOR=wandb
echo $PYTHONPATH
export  ROLE=master
export INDEX=0
#CUDA_VISIBLE_DEVICES="0" python test_delete_nmp.py 0 
CUDA_VISIBLE_DEVICES="0" python test_multi_worker.py  --category_num=1000 \
                          --per_device_train_batch_size=100 --weight_decay=0.0001 --label_smoothing=0.1 --train_num=1300 \
                          --test_num=300 --learning_rate=0.05 --minimum_learning_rate=0.0001 \
                          --save_steps=2 --data="/data/resnet50/extract" --num_train_epochs=5 --warmup_steps=5

#python test_multi_worker.py  0
