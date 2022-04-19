import os
import tensorflow as tf
import config as c
from tqdm import tqdm
from utils.data_utils import test_iterator
from utils.eval_utils import cross_entropy_batch, correct_num_batch, l2_loss
from model.ViT import ViT
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

@tf.function
def test_step(model, images, labels):
    prediction = model(images, training=False)
    ce = cross_entropy_batch(labels, prediction)
    return ce, prediction

def test(model, log_file):
    data_iterator = test_iterator()

    sum_ce = 0
    sum_correct_num = 0

    for i in tqdm(range(c.test_iterations)):
        images, labels = data_iterator.next()
        ce, prediction = test_step(model, images, labels)
        correct_num = correct_num_batch(labels, prediction)

        sum_ce += ce * c.batch_size
        sum_correct_num += correct_num
        print('ce: {:.4f}, accuracy: {:.4f}'.format(ce, correct_num / c.batch_size))

    log_file.write('test: cross entropy loss: {:.4f}, accuracy: {:.4f}\n'.format(sum_ce / c.test_num, sum_correct_num / c.test_num))

if __name__ == '__main__':
    # gpu config
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(device=physical_devices[0], enable=True)

    # get model
    model_config = {"image_size":224,
                "patch_size":16,
                "num_classes":1000,
                "dim":384,
                "depth":12,
                "heads":6,
                "mlp_dim":1024,
                "hidden_layer_shape": 1536}

    # get model
    model = ViT(**model_config)

    # show
    model.build(input_shape=(None,) + c.input_shape)

    if c.load_weight_file is None:
        print('Please fill in the path of model weight in config.py')
    else:
        model.load_weights(c.load_weight_file)

    # test
    with open('result/log/test_log.txt', 'a') as f:
        test(model, f)
