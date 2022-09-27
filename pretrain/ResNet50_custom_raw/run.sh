rm -rf checkpoint
export PLATFORM=pytorch 
export JOBID=f31bf13b-c0aa-4ff5-8882-805e4e94c6f5
export USE_DDP=1
export RANK=0
export LOCAL_RANK=0
export PLATFORM=tensorflow
#export LD_LIBRARY_PATH=/home/protago/miniconda3/envs/netmind-tf2/lib:${LD_LIBRARY_PATH}
export PYTHONPATH=/home/protago/ly/NetMind-Images/NetmindModelEnv/NetmindMixins:${PYTHONPATH}
#export TIME_ESTIMATE_PROCESS=1
export MONITOR=wandb
#python test_delete_nmp.py
python test.py 
