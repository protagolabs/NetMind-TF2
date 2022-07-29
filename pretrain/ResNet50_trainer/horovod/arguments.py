import os
import argparse
import os


def setup_args():
    """
    设置训练参数
    """
    parser = argparse.ArgumentParser(description='Keras ImageNet Example',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--train-dir', default=os.path.expanduser('~/imagenet/train'),
                        help='path to training data')
    parser.add_argument('--val-dir', default=os.path.expanduser('~/imagenet/validation'),
                        help='path to validation data')
    parser.add_argument('--log-dir', default='./logs',
                        help='tensorboard log directory')
    parser.add_argument('--checkpoint-format', default='./checkpoint-{epoch}.h5',
                        help='checkpoint file format')
    parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                        help='use fp16 compression during allreduce')

    # Default settings from https://arxiv.org/abs/1706.02677.

    parser.add_argument('--val-batch-size', type=int, default=32,
                        help='input batch size for validation')
    parser.add_argument('--epochs', type=int, default=90,
                        help='number of epochs to train')

    parser.add_argument('--warmup-epochs', type=float, default=5,
                        help='number of warmup epochs')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum')
    parser.add_argument('--wd', type=float, default=0.00005,
                        help='weight decay')

    #netind related argument
    parser.add_argument('--learning_rate', type=float, default=0.0125,
                        help='learning rate for a single GPU')
    parser.add_argument('--num_train_epochs', type=int, default=90,
                        help='number of epochs to train')
    parser.add_argument('--per_device_train_batch_size', type=int, default=16, required=False,
                        help='input batch size for training')
    parser.add_argument('--output_dir', default='model_1', type=str, required=False, help='')
    parser.add_argument('--save_epoch', default=2, type=int, required=False, help='')
    parser.add_argument('--save_steps', default=5000, type=int, required=False, help='')


    args = parser.parse_args()

    return args