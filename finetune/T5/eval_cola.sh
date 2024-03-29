


# $DATA_DIR = "/home/protago/Xing/datasets/glue_data/MRPC"
# MODEL_DIR="/home/protago/Xing/finetuneT5/text-to-text-transfer-transformer/pretrained_models/base"

MODEL_DIR="./finetune_cola"
# MODEL_DIR="gs://t5-data/pretrained_models/base"
# PRETRAINED_STEPS=999900
# FINETUNE_STEPS=100000 # 262,144
# CHECKPOINT_STEPS=4000


# t5_mesh_transformer  \
#   --pretrained_model_dir="${MODEL_DIR}" \
#   --model_dir="./finetune_mrpc/" \
#   --gin_param="utils.run.mesh_shape = 'model:1,batch:1'" \
#   --gin_param="utils.run.mesh_devices = ['gpu:0']" \
#   --gin_param="MIXTURE_NAME = 'glue_mrpc_v002'" \
#   --gin_param="utils.run.train_steps=$((PRETRAINED_STEPS+FINETUNE_STEPS))" \
#   --gin_param="utils.run.save_checkpoints_steps=$((CHECKPOINT_STEPS))" \
#   --gin_file="${MODEL_DIR}/operative_config.gin" \
#   --gin_param="utils.run.batch_size = ('tokens_per_batch', 2048)" \

# t5_mesh_transformer \
#   --model_dir="${MODEL_DIR}" \
#   --gin_param="utils.run.mesh_shape = 'model:1,batch:1'" \
#   --gin_param="utils.run.mesh_devices = ['gpu:0']" \
#   --gin_file="${MODEL_DIR}/operative_config.gin" \
#   --gin_file="eval.gin" \
#   --gin_file="beam_search.gin" \
#   --gin_param="run.dataset_split = 'validation'" \
#   --gin_param="MIXTURE_NAME = 'glue_stsb_v002'" \
#   --gin_param="eval_checkpoint_step = 'all'" \
#   --gin_param="utils.run.batch_size = ('tokens_per_batch', 2048)"

t5_mesh_transformer \
  --model_dir="${MODEL_DIR}" \
  --gin_param="utils.run.mesh_shape = 'model:1,batch:1'" \
  --gin_param="utils.run.mesh_devices = ['gpu:0']" \
  --gin_file="${MODEL_DIR}/operative_config.gin" \
  --gin_file="eval.gin" \
  --gin_file="beam_search.gin" \
  --gin_param="run.dataset_split = 'validation'" \
  --gin_param="MIXTURE_NAME = 'glue_cola_v002'" \
  --gin_param="eval_checkpoint_step = 'all'" \
  --gin_param="utils.run.batch_size = ('tokens_per_batch', 2048)"