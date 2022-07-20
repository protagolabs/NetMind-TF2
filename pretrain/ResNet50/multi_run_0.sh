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
export MONITOR=tensorboard
echo $PYTHONPATH
export  ROLE=master
export INDEX=0
#CUDA_VISIBLE_DEVICES="0" python test_delete_nmp.py 0 
CUDA_VISIBLE_DEVICES="0" python test_multi_worker.py  --n_workers=2  --category_num=1000 \
                          --batch_size=100 --weight_decay=0.0001 --label_smoothing=0.1 --train_num=2600 \
                          --test_num=260 --initial_learning_rate=0.05 --minimum_learning_rate=0.0001 \
                          --save_steps=2 --data="/data/resnet50/extracted_2600"

#python test_multi_worker.py  0
