

TASK_NAME="rte"
MODEL_DIR="./finetune_${TASK_NAME}"

t5_mesh_transformer \
  --model_dir="${MODEL_DIR}" \
  --gin_param="utils.run.mesh_shape = 'model:1,batch:1'" \
  --gin_param="utils.run.mesh_devices = ['gpu:0']" \
  --gin_file="${MODEL_DIR}/operative_config.gin" \
  --gin_file="eval.gin" \
  --gin_file="beam_search.gin" \
  --gin_param="run.dataset_split = 'validation'" \
  --gin_param="MIXTURE_NAME = 'glue_${TASK_NAME}_v002'" \
  --gin_param="eval_checkpoint_step = 'all'" \
  --gin_param="utils.run.batch_size = ('tokens_per_batch', 2048)"