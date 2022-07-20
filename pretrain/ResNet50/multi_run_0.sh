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
export  ROLE=MASTER
#CUDA_VISIBLE_DEVICES="0" python test_delete_nmp.py 0 
CUDA_VISIBLE_DEVICES="0" python test_multi_worker.py  0
#python test_multi_worker.py  0
