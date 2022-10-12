import argparse
import os


def setup_args():
    """
    设置训练参数
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', default= 'roberta-base' , type=str, required=False, help='')
    parser.add_argument('--do_train', default= True , type=bool, required=False, help='')
    parser.add_argument('--data', default=os.getenv("DATA_LOCATION"), type=str, required=False, help='')

    # adv
    #param below belong to resnet50 trainer
    parser.add_argument("--learning_rate", default=0.05, type=float)
    parser.add_argument("--num_train_epochs", default=10, type=int)
    parser.add_argument('--per_device_train_batch_size', default=100, type=int, required=False, help='')

    return parser.parse_args()
