import os
import tensorflow as tf
import config as c
from tqdm import tqdm
from utils.data_utils_mm import train_iterator, test_iterator
from utils.eval_utils import cross_entropy_batch, correct_num_batch, l2_loss
from model.ResNet import ResNet


physical_devices = tf.config.list_physical_devices('GPU') 
for gpu_instance in physical_devices: 
    tf.config.experimental.set_memory_growth(gpu_instance, True)



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
            ce = tf.keras.losses.categorical_crossentropy(labels, prediction, label_smoothing=c.label_smoothing)
            l2 = l2_loss(model)
            # loss = ce + l2
            loss = ce

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(list(zip(grads, model.trainable_variables)))
            return ce, l2

    per_replica_ce,  per_replica_l2 = mirrored_strategy.run(step_fn, args=(dist_inputs,))
    # mean_loss = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
    mean_ce = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_ce, axis=None)
    # mean_l2 = mirrored_strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_l2, axis=None)


    return mean_ce
    

@tf.function
def test_step(dist_inputs):

    def step_fn(inputs):
        images, labels = inputs

        prediction = model(images, training=False)
        ce = cross_entropy_batch(labels, prediction)
        correct_num = correct_num_batch(labels, prediction)

        return ce, correct_num

    per_replica_losses, per_replica_pred = mirrored_strategy.run(step_fn, args=(dist_inputs,))
    mean_loss = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
    acc = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_pred, axis=None)
    return mean_loss, acc

def set_input_shape(img, label):
    img = img.set_shape(c.input_shape)
    label = label.set_shape([c.category_num])
    return img, label

## Using `tf.distribute.Strategy` with custom training loops

if __name__ == '__main__':

    mirrored_strategy = tf.distribute.MirroredStrategy()
    # mirrored_strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1", "GPU:2", "GPU:3"])
    # mirrored_strategy = tf.distribute.OneDeviceStrategy(device="GPU:0")


    num_gpus = mirrored_strategy.num_replicas_in_sync

    print('Number of devices: {}'.format(num_gpus))

    global_batch_size = c.batch_size *  num_gpus

# First, we create the model and optimizer inside the strategy's scope. This ensures that any variables created with the model and optimizer are mirrored variables.

    with mirrored_strategy.scope():

        model = ResNet(50)

        model.build(input_shape=(None,) + c.input_shape)
        
        model.summary()

        inputs = tf.keras.Input(shape=c.input_shape)


        # load pretrain
        if c.load_weight_file is not None:
            model.load_weights(c.load_weight_file)
            # print('pretrain weight l2 loss:{:.4f}'.format(l2_loss(model)))

        # here we automatically change the iterations per epoch based on number of gpus
        learning_rate_schedules = CosineDecayWithWarmUP(initial_learning_rate=c.initial_learning_rate * num_gpus,
                                                        decay_steps=c.epoch_num * int(c.iterations_per_epoch / num_gpus)  - int(c.warm_iterations / num_gpus),
                                                        alpha=c.minimum_learning_rate * num_gpus,
                                                        warm_up_step=int(c.warm_iterations / num_gpus))

        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate_schedules, momentum=0.9)



    
# Next, we create the input dataset and call `tf.distribute.Strategy.experimental_distribute_dataset` to distribute the dataset based on the strategy.


    # dataset = train_iterator().batch(global_batch_size)
    # train_data_iterator = mirrored_strategy.experimental_distribute_dataset(dataset) 
    # for inputs in train_data_iterator:
    #     print(train_step(inputs))
    #     # train_step(inputs)
    # train_iterator = train_iterator.map(set_input_shape)
    dataset_train = train_iterator().batch(global_batch_size)
    train_data_iterator = iter(mirrored_strategy.experimental_distribute_dataset(dataset_train))


    #  eval
    dataset_eval = test_iterator().batch(global_batch_size)
    test_data_iterator = iter(mirrored_strategy.experimental_distribute_dataset(dataset_eval))


    with open(c.log_file, 'a') as f:

        for epoch_num in range(c.epoch_num):

            # train 
            sum_ce = 0
            for i in tqdm(range(int(c.train_num // global_batch_size))):
            # for ds in train_data_iterator:
                ds = train_data_iterator.get_next()
                # print(ds)
                loss_ce = train_step(ds)
                sum_ce += tf.reduce_sum(loss_ce)
                
                # print(loss_ce)
                # print('ce: {:.4f}'.format(tf.reduce_mean(loss_ce)))
            print('train: cross entropy loss: {:.4f}\n'.format(sum_ce / c.train_num))
            f.write('train: cross entropy loss: {:.4f}\n'.format(sum_ce / c.train_num))

            # validate
            sum_ce = 0
            sum_correct_num = 0
            for i in tqdm(range(int(c.test_num // global_batch_size))):
            # for ds in test_data_iterator:
                ds = test_data_iterator.get_next()
                ce, correct_num = test_step(ds)
                sum_ce += tf.reduce_sum(ce)
                sum_correct_num +=  tf.reduce_sum(correct_num)

            print('test: cross entropy loss: {:.4f}, accuracy: {:.4f}\n'.format(sum_ce / c.test_num, sum_correct_num / c.test_num))
            f.write('test: cross entropy loss: {:.4f}, accuracy: {:.4f}\n'.format(sum_ce / c.test_num, sum_correct_num / c.test_num))


            model.save_weights(c.save_weight_file, save_format='h5')

            # save intermediate results
            if epoch_num % 5 == 4:
                os.system('cp {} {}_epoch_{}.h5'.format(c.save_weight_file, c.save_weight_file.split('.')[0], epoch_num))

"""In the example above, we iterated over the `dist_dataset` to provide input to your training. We also provide the  `tf.distribute.Strategy.make_experimental_numpy_dataset` to support numpy inputs. You can use this API to create a dataset before calling `tf.distribute.Strategy.experimental_distribute_dataset`.
Another way of iterating over your data is to explicitly use iterators. You may want to do this when you want to run for a given number of steps as opposed to iterating over the entire dataset.
The above iteration would now be modified to first create an iterator and then explicity call `next` on it to get the input data.
"""

