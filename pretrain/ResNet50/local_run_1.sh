rm -rf checkpoint
export PLATFORM=tensorflow
export JOBID=f31bf13b-c0aa-4ff5-8882-805e4e94c6f5
export USE_DDP=1
export RANK=0
export LOCAL_RANK=0
export PLATFORM=tensorflow
#export LD_LIBRARY_PATH=/home/protago/miniconda3/envs/netmind-tf2/lib:${LD_LIBRARY_PATH}
export PYTHONPATH=/home/protago/ly/NetMind-Images/NetmindModelEnv/NetmindMixins:${PYTHONPATH}
#export TIME_ESTIMATE_PROCESS=1
export MONITOR=tensorboard
#python test_delete_nmp.py
echo $PYTHONPATH
#CUDA_VISIBLE_DEVICES="1" python test_delete_nmp.py 1
CUDA_VISIBLE_DEVICES="1" python test_multi_worker.py 1
#python test_multi_worker.py 1
