rm -rf checkpoint
export PLATFORM=tensorflow
export JOBID=f31bf13b-c0aa-4ff5-8882-805e4e94c6f5
export USE_DDP=1
export RANK=0
export LOCAL_RANK=0
export PLATFORM=tensorflow
#export LD_LIBRARY_PATH=/home/protago/miniconda3/envs/netmind-tf2/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=/usr/lib64:${LD_LIBRARY_PATH}
export PYTHONPATH=/home/protago/ly/NetMind-Images/NetmindModelEnv/NetmindMixins:${PYTHONPATH}
#export TIME_ESTIMATE_PROCESS=1
export MONITOR=wandb
#python test_delete_nmp.py
export  INDEX=1
#CUDA_VISIBLE_DEVICES="0" python test_delete_nmp.py 0
CUDA_VISIBLE_DEVICES="0" python test_multi_worker.py  --n_workers=2  --category_num=1000 \
                          --batch_size=100 --weight_decay=0.0001 --label_smoothing=0.1 --train_num=2600 \
                          --test_num=260 --initial_learning_rate=0.05 --minimum_learning_rate=0.0001 \
                          --save_steps=10 --data="/data/resnet50/extracted_2600" epoch_num=5
