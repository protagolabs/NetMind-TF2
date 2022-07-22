import os
import json
import time
import traceback
import tensorflow as tf
from  keras.backend import set_session
import config as c
from tqdm import tqdm
from utils.data_utils_mm import train_iterator, test_iterator, cnt
from utils.eval_utils import cross_entropy_batch, correct_num_batch, l2_loss
from model.ResNet import ResNet
from NetmindMixins.Netmind import nmp, NetmindDistributedModel, NetmindOptimizer, NetmindDistributedModel
import sys
from arguments import setup_args

args = setup_args()

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
physical_devices = tf.config.list_physical_devices('GPU')
for gpu_instance in physical_devices:
    tf.config.experimental.set_memory_growth(gpu_instance, True)

def allow_memory_growth():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(f'{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs')
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    return


class CosineDecayWithWarmUP(tf.keras.experimental.CosineDecay):
    def __init__(self, initial_learning_rate, decay_steps, alpha=0.0, warm_up_step=0, name=None):
        self.warm_up_step = warm_up_step
        super(CosineDecayWithWarmUP, self).__init__(initial_learning_rate=initial_learning_rate,
                                                    decay_steps=decay_steps,
                                                    alpha=alpha,
                                                    name=name)

    @tf.function
    def __call__(self, step):
        if step <= self.warm_up_step:
            return tf.cast(step / self.warm_up_step * self.initial_learning_rate, tf.float32)
        else:
            return super(CosineDecayWithWarmUP, self).__call__(step - self.warm_up_step)



'''
we define one step of the training. We will use `tf.GradientTape` to compute gradients and optimizer to apply those gradients to
update our model's variables. To distribute this training step, we put in in a function `step_fn` and pass it to
`tf.distrbute.Strategy.experimental_run_v2` along with the dataset inputs that we get from `dist_dataset` created before:
'''

@tf.function
def train_step(dist_inputs):

    def step_fn(inputs):
        images, labels = inputs

        with tf.GradientTape() as tape:
            prediction = model(images)
            # ce = cross_entropy_batch(labels, prediction, label_smoothing=c.label_smoothing)
            ce = tf.keras.losses.categorical_crossentropy(labels, prediction, label_smoothing=args.label_smoothing)
            l2 = l2_loss(model)
            # loss = ce + l2
            loss = ce

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(list(zip(grads, model.trainable_variables)))
            return ce, l2

    per_replica_ce,  per_replica_l2 = multi_worker_mirrored_strategy.run(step_fn, args=(dist_inputs,))
    # mean_loss = multi_worker_mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
    mean_ce = multi_worker_mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_ce, axis=None)
    # mean_l2 = multi_worker_mirrored_strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_l2, axis=None)


    return mean_ce


@tf.function
def test_step(dist_inputs):

    def step_fn(inputs):
        images, labels = inputs

        prediction = model(images, training=False)
        ce = cross_entropy_batch(labels, prediction)
        correct_num = correct_num_batch(labels, prediction)

        return ce, correct_num

    per_replica_losses, per_replica_pred = multi_worker_mirrored_strategy.run(step_fn, args=(dist_inputs,))
    mean_loss = multi_worker_mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
    acc = multi_worker_mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, tf.compat.v1.to_float(per_replica_pred, name='ToFloat'), axis=None)
    return mean_loss, acc

def set_input_shape(img, label):
    img = img.set_shape(c.input_shape)
    label = label.set_shape([args.category_num])
    return img, label

## Using `tf.distribute.Strategy` with custom training loops

if __name__ == '__main__':
    
    begin = time.time()
    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())
    if not os.getenv('TF_CONFIG'):
        c.tf_config['task']['index'] = int(os.getenv('INDEX'))
        os.environ['TF_CONFIG'] = json.dumps(c.tf_config)
    print(os.environ['TF_CONFIG'], type(os.environ['TF_CONFIG']))
    res = json.loads(os.environ['TF_CONFIG'])
    print(res, type(res))
    print(res['task']['index'])
    print('TF_CONFIG : ', os.environ['TF_CONFIG'])

    #allow_memory_growth must be called after  tf.distribute.MultiWorkerMirroredStrategy
    multi_worker_mirrored_strategy = tf.distribute.MultiWorkerMirroredStrategy()
    # multi_worker_mirrored_strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1", "GPU:2", "GPU:3"])
    #multi_worker_mirrored_strategy = tf.distribute.OneDeviceStrategy(device="GPU:0")


    num_gpus = multi_worker_mirrored_strategy.num_replicas_in_sync

    print('Number of devices: {}'.format(num_gpus))
    nmp.init(load_checkpoint=False)

    n_workers = len(json.loads(os.environ['TF_CONFIG']).get('cluster', {}).get('worker'))
    global_batch_size = args.per_device_train_batch_size *  n_workers

# First, we create the model and optimizer inside the strategy's scope. This ensures that any variables created with the model and optimizer are mirrored variables.

    with multi_worker_mirrored_strategy.scope():

        model = ResNet(50)
        model.build(input_shape=(None,) + c.input_shape)
        model.summary()
        inputs = tf.keras.Input(shape=c.input_shape)

        """
        # load pretrain
        if c.load_weight_file is not None:
            model.load_weights(c.load_weight_file)
            # print('pretrain weight l2 loss:{:.4f}'.format(l2_loss(model)))
        """

        iterations_per_epoch = int(args.train_num / args.per_device_train_batch_size)
        warm_iterations = iterations_per_epoch
        # here we automatically change the iterations per epoch based on number of gpus
        learning_rate_schedules = CosineDecayWithWarmUP(initial_learning_rate=args.learning_rate * n_workers,
                                                        decay_steps=args.num_train_epochs * int(iterations_per_epoch / n_workers)  - int(warm_iterations / n_workers),
                                                        alpha=args.minimum_learning_rate * n_workers,
                                                        warm_up_step=int(args.warmup_steps))
                                                        #warm_up_step=int(warm_iterations / n_workers))

        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate_schedules, momentum=0.9)
        print('scope end')



    # Next, we create the input dataset and call `tf.distribute.Strategy.experimental_distribute_dataset` to distribute the dataset based on the strategy.


        # dataset = train_iterator().batch(global_batch_size)
        # train_data_iterator = multi_worker_mirrored_strategy.experimental_distribute_dataset(dataset)
        # for inputs in train_data_iterator:
        #    print(train_step(inputs))
        #    # train_step(inputs)
        # train_iterator = train_iterator.map(set_input_shape)
        train_data_path = args.data + "/train"
        dataset_train = train_iterator(train_data_path=train_data_path).batch(global_batch_size)
        train_data_iterator = iter(multi_worker_mirrored_strategy.experimental_distribute_dataset(dataset_train))


        NetmindDistributedModel(model)
        #  eval
        test_data_path = args.data + "/val"
        dataset_eval = test_iterator(test_data_path=test_data_path).batch(global_batch_size)
        test_data_iterator = iter(multi_worker_mirrored_strategy.experimental_distribute_dataset(dataset_eval))

        nmp.init_train_bar(total_epoch=args.num_train_epochs, step_per_epoch=args.train_num//args.per_device_train_batch_size)

        t_total = nmp.cur_step
        epochs_trained = nmp.cur_epoch
        print(f'epochs_trained: {epochs_trained}')
        print(f'test.py pid : {os.getpid()} ')
        next_cnt = 0
        with open(c.log_file, 'a') as f:

            for epoch_num in range(epochs_trained, args.num_train_epochs):
                print(f'training with epoch_num : {epoch_num} in range of  {epochs_trained}-------{args.num_train_epochs}')

                # train
                sum_ce = 0
                print(f'in one epoch : loop start in range {args.train_num // global_batch_size}')
                for i in tqdm(range(int(args.train_num // global_batch_size))):
                    if nmp.should_skip_step():
                        continue
                    # for ds in train_data_iterator:
                    ds = train_data_iterator.get_next()
                    print(f'len ds : {len(ds)}, type : {type(ds)}')
                    next_cnt += 1
                    loss_ce = train_step(ds)
                    sum_ce += tf.reduce_sum(loss_ce)
                    # netmind relatived
                    #nmp.step({"loss": loss_ce, "Learning rate": scheduler.get_last_lr()[0]})
                    learing_rate = learning_rate_schedules(i)
                    print(type(learing_rate.numpy()), learing_rate.numpy(), f'current  epoch : {epoch_num}')
                    final_loss = float((sum_ce / args.train_num).numpy())
                    final_learing_rate = float(learing_rate.numpy())
                    print(f"loss : {final_loss}, type: {type(final_loss)}")
                    print(f"learing_rate : {final_learing_rate}, type: {type(final_learing_rate)}")
                    nmp.step({"loss": final_loss, "Learning rate":final_learing_rate})
                    print(f'save_pretrained_by_step : {args.save_steps}')
                    nmp.save_pretrained_by_step(args.save_steps)


                print('train: cross entropy loss: {:.4f}\n'.format(sum_ce / args.train_num))
                f.write('train: cross entropy loss: {:.4f}\n'.format(sum_ce / args.train_num))

                # validate
                sum_ce = 0
                sum_correct_num = 0
                for i in tqdm(range(int(args.test_num // global_batch_size))):
                    ds = test_data_iterator.get_next()
                    ce, correct_num = test_step(ds)
                    sum_ce += tf.reduce_sum(ce)
                    sum_correct_num +=  tf.reduce_sum(correct_num)

                print('test: cross entropy loss: {:.4f}, accuracy: {:.4f}\n'.format(sum_ce / args.test_num, sum_correct_num / args.test_num))
                f.write('test: cross entropy loss: {:.4f}, accuracy: {:.4f}\n'.format(sum_ce / args.test_num, sum_correct_num / args.test_num))

                #TODO: netmind already save weights
                #model.save_weights(c.save_weight_file, save_format='h5')

                # save intermediate results
                #if epoch_num % 5 == 4:
                #    os.system('cp {} {}_epoch_{}.h5'.format(c.save_weight_file, c.save_weight_file.split('.')[0], epoch_num))
            nmp.finish_training()
            end = time.time()
    
            print(f'training finished... cnt : {cnt}, next_cnt : {next_cnt}, cost : {end-begin}')

        """In the example above, we iterated over the `dist_dataset` to provide input to your training. We also provide the  `tf.distribute.Strategy.make_experimental_numpy_dataset` to support numpy inputs. You can use this API to create a dataset before calling `tf.distribute.Strategy.experimental_distribute_dataset`.
        Another way of iterating over your data is to explicitly use iterators. You may want to do this when you want to run for a given number of steps as opposed to iterating over the entire dataset.
        The above iteration would now be modified to first create an iterator and then explicity call `next` on it to get the input data.
        """
