import argparse
import os


def setup_args():
    """
    设置训练参数
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=os.getenv("DATA_LOCATION"), type=str, required=False, help='')


    #roberta
    parser.add_argument("--save_steps", default=20, type=int)
    parser.add_argument("--num_train_epochs", default=5, type=int)
    parser.add_argument('--per_device_train_batch_size', default=16, type=int, required=False, help='')

    parser.add_argument("--learning_rate", default=0.0001, type=float)
    parser.add_argument("--adam_beta1", default=0.9, type=float)
    parser.add_argument("--adam_beta2", default=0.98, type=float)
    parser.add_argument("--adam_epsilon", default=1e-6, type=float)
    parser.add_argument('--weight_decay', default=0.01, type=float, required=True, help='')
    parser.add_argument("--warmup_proportion", default=0.1, type=float)

    parser.add_argument("--config_name", default="roberta-base", type=str)
    parser.add_argument("--tokenizer_name", default="roberta-base", type=str)
    #parser.add_argument('--model_name_or_path', default='roberta-base', type=str, required=False, help='')

    return parser.parse_args()
