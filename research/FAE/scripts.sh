# python main.py --data="/home/xing/datasets/CelebA/"



# python main_pl.py fit --model.data_path="/home/xing/datasets/CelebA/" --trainer.gpus=4 \
# --trainer.logger=TensorBoardLogger \
# --trainer.logger.save_dir="tb_logs" \
# --trainer.logger.name="resnet50_pretrain" \
# --trainer.logger.version="find_lr" \
# --trainer.auto_lr_find=True \
# --print_config > resnet50_config.yaml

# python main_pl.py fit --config resnet50_config.yaml

# python main_pl.py test --config ./resnet50_config.yaml 


# python main_vit.py fit --model.data_path ../face_editing/datasets/CelebA/ --trainer.gpus=4 \
# --trainer.logger=TensorBoardLogger \
# --trainer.logger.save_dir="tb_logs" \
# --trainer.logger.name="vit_large_patch16_pretrain" \
# --trainer.logger.version="find_lr" \
# --trainer.auto_lr_find=True \
# --print_config > vit_large_patch16_config.yaml

# python main_vit.py fit --config vit_large_patch16_config.yaml
# python main_vit.py test --config ./vit_large_patch16_config.yaml

# python main_vit.py fit \
# --model.data_path ../face_editing/datasets/CelebA/ \
# --model.arch="vit_base_patch16" \
# --model.batch_size=512 \
# --model.lr=0.001 \
# --model.warmup_epochs=40 \
# --trainer.gpus=4 \
# --trainer.logger=TensorBoardLogger \
# --trainer.logger.save_dir="tb_logs" \
# --trainer.logger.name="vit_base_patch16_pretrain" \
# --trainer.logger.version="find_lr" \
# --trainer.auto_lr_find=True \
# --trainer.max_epochs=200 \
# --print_config > vit_base_patch16_config.yaml

# python main_vit.py fit \
# --model.data_path ../face_editing/datasets/CelebA/ \
# --model.arch="vit_base_patch16" \
# --model.batch_size=128 \
# --model.lr=0.001 \
# --model.min_lr=1e-6 \
# --model.warmup_epochs=5 \
# --model.pretrained=True \
# --trainer.gpus=4 \
# --trainer.logger=TensorBoardLogger \
# --trainer.logger.save_dir="tb_logs" \
# --trainer.logger.name="vit_base_patch16_finetune" \
# --trainer.logger.version="find_lr" \
# --trainer.auto_lr_find=True \
# --trainer.max_epochs=50 \
# --print_config > vit_base_patch16_ft_config.yaml


# python main_vit.py fit \
# --model.data_path ../face_editing/datasets/CelebA/ \
# --model.arch="vit_large_patch16" \
# --model.batch_size=128 \
# --model.lr=0.001 \
# --model.min_lr=1e-6 \
# --model.warmup_epochs=5 \
# --model.pretrained=True \
# --trainer.gpus=4 \
# --trainer.logger=TensorBoardLogger \
# --trainer.logger.save_dir="tb_logs" \
# --trainer.logger.name="vit_large_patch16_finetune" \
# --trainer.logger.version="find_lr" \
# --trainer.auto_lr_find=True \
# --trainer.max_epochs=50 \
# --print_config > vit_large_patch16_ft_config.yaml


python main_vit.py fit \
--model.data_path="/home/xing/datasets/CelebA/" \
--model.arch="vit_base_patch16" \
--model.batch_size=128 \
--model.lr=0.001 \
--model.min_lr=1e-6 \
--model.warmup_epochs=5 \
--model.pretrained=False \
--trainer.gpus=4 \
--trainer.logger=TensorBoardLogger \
--trainer.logger.save_dir="tb_logs" \
--trainer.logger.name="vit_base_patch16_finetune_celeba_pretrain" \
--trainer.logger.version="find_lr" \
--trainer.auto_lr_find=True \
--trainer.max_epochs=50 \
--print_config > vit_base_patch16_ft_celeba_config.yaml