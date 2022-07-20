import argparse
import os


def setup_args():
    """
    设置训练参数
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_workers', default=1, type=int, required=True, help='')
    parser.add_argument('--category_num', default=1000, type=int, required=False, help='')
    parser.add_argument('--batch_size', default=100, type=int, required=False, help='')
    parser.add_argument('--input_shape', default=(), type=tuple, required=False, help='')
    parser.add_argument('--weight_decay', default= 1e-4, type=float, required=False, help='')
    parser.add_argument('--label_smoothing', default=0.1, type=float, required=False, help='')
    parser.add_argument('--train_num', default=100, type=int, required=False, help='')
    parser.add_argument("--test_num", help='use distributed training')
    # adv
    parser.add_argument("--initial_learning_rate", default=0.05, type=float)
    parser.add_argument("--minimum_learning_rate", default=0.0001, type=float)
    parser.add_argument("--log_file", default=str, type=str)
    parser.add_argument("--save_steps", default=100, type=int)

    return parser.parse_args()