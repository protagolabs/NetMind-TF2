import os
import json
import time
import traceback
import tensorflow as tf
from  keras.backend import set_session
import config as c
from tqdm import tqdm
import  logging
from utils.data_utils_mm import train_iterator, test_iterator, cnt
from utils.eval_utils import cross_entropy_batch, correct_num_batch, l2_loss
from model.ResNet import ResNet
from NetmindMixins.Netmind import nmp, NetmindDistributedModel, NetmindOptimizer, NetmindDistributedModel
import sys
from arguments import setup_args
logger = logging.getLogger()
formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(filename)s:%(lineno)d %(message)s', '%Y-%m-%d %H:%M:%S')

console_handler = logging.StreamHandler(sys.stdout)
console_handler.formatter = formatter
logger.addHandler(console_handler)
logger.setLevel(logging.INFO)

args = setup_args()


class SavePretrainedCallback(tf.keras.callbacks.Callback):
    # Hugging Face models have a save_pretrained() method that saves both the weights and the necessary
    # metadata to allow them to be loaded as a pretrained model in future. This is a simple Keras callback
    # that saves the model with this method after each epoch.
    def __init__(self, **kwargs):
        super().__init__()
        self.epoch = 0

    def on_train_begin(self, logs=None):
        logger.info(f'on_train_begin : log : {logs}')
        # here we init train&eval bar
        nmp.init(load_checkpoint=False)
        NetmindDistributedModel(self.model)
        nmp.init_train_bar(total_epoch=args.num_train_epochs, step_per_epoch=train_num  // global_batch_size)
        nmp.init_eval_bar(total_epoch=args.num_train_epochs)
        epochs_trained = nmp.cur_epoch
        logger.info(f'epochs_trained: {epochs_trained}')

    def on_train_end(self, logs=None):
        logger.info(f'finish trianing log : {logs}')
        nmp.finish_training()
    
    def on_train_batch_begin(self, batch, logs=None):
        logger.info(f'batch : {batch}, logs: {logs}')
        if nmp.should_skip_step():
            return

    def on_train_batch_end(self, batch, logs=None):
        logger.info(f'on_train_batch_end : batch : {batch} , log : {logs}')
        print(f'type: {self.model.optimizer.iterations}')
        """
        learning_rate = self.model.optimizer.learning_rate(self.model.optimizer.iterations.numpy())

        learning_rate = tf.keras.backend.get_value(learning_rate)
        """
        learning_rate = self.model.optimizer.learning_rate.numpy()
        logger.info(f'learning_rate : {learning_rate}')

        nmp.step({"loss": float(logs['loss']),
                  "Learning rate": float(learning_rate)})
        logger.info(f'save_pretrained_by_step : {args.save_steps}')
        nmp.save_pretrained_by_step(args.save_steps)

    def on_test_end(self, logs=None):
        logger.info(f'log : {logs}')
        nmp.evaluate(logs)
        


if __name__ == '__main__':

    """
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.4
    #config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    gpu_list = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpu_list:
        tf.config.experimental.set_memory_growth(gpu, True)
    """

    if not os.getenv('TF_CONFIG'):
        c.tf_config['task']['index'] = int(os.getenv('INDEX'))
        os.environ['TF_CONFIG'] = json.dumps(c.tf_config)

    multi_worker_mirrored_strategy = tf.distribute.MultiWorkerMirroredStrategy()

    num_gpus = multi_worker_mirrored_strategy.num_replicas_in_sync
    print('Number of devices: {}'.format(num_gpus))

    n_workers = len(json.loads(os.environ['TF_CONFIG']).get('cluster', {}).get('worker'))
    global_batch_size = args.per_device_train_batch_size * n_workers

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        args.data,
        validation_split=0.2,
        subset="training",
        seed=1337,
        image_size=c.input_shape[:2],
        batch_size=global_batch_size,
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        args.data,
        validation_split=0.2,
        subset="validation",
        seed=1337,
        image_size=c.input_shape[:2],
        batch_size=global_batch_size,
    )

    train_num = len(train_ds.file_paths)
    test_num = len(val_ds.file_paths)
    category_num = len(train_ds.class_names)

    train_ds = train_ds.cache().repeat().prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.cache()
# First, we create the model and optimizer inside the strategy's scope. This ensures that any variables created with the model and optimizer are mirrored variables.

    with multi_worker_mirrored_strategy.scope():

        # model = ResNet(50)
        # model = ResNetTypeII(layer_params=[3, 4, 6, 3], input_shape=c.input_shape)

        # model.build(input_shape=(None,) + c.input_shape)
        
        # model.summary()
        # print('initial l2 loss:{:.4f}'.format(l2_loss(model)))

        inputs = tf.keras.Input(shape=c.input_shape)

        outputs = tf.keras.applications.resnet50.ResNet50(  # Add the rest of the model
            weights=None, input_shape=c.input_shape, classes=category_num, classifier_activation="softmax"
        )(inputs)

        model = tf.keras.Model(inputs, outputs)

        model.summary()

        optimizer = tf.keras.optimizers.Adam(args.learning_rate *  num_gpus)

        model.compile(
            optimizer=optimizer,
            loss=tf.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=tf.keras.metrics.SparseCategoricalAccuracy()
        )
        

        

    
# Next, we create the input dataset and call `tf.distribute.Strategy.experimental_distribute_dataset` to distribute the dataset based on the strategy.

    train_data_iterator = multi_worker_mirrored_strategy.experimental_distribute_dataset(train_ds)


    #  eval
    # dataset_eval = test_iterator().batch(global_batch_size, drop_remainder=False)
    test_data_iterator = multi_worker_mirrored_strategy.experimental_distribute_dataset(val_ds)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir="tb_logs/resnet50",
        histogram_freq=0,
        write_graph=True,
        write_images=False,
        write_steps_per_second=False,
        update_freq="epoch",
        profile_batch=0,
        embeddings_freq=0,
        embeddings_metadata=None,
    )

    #### we add the nmp callback here ###

    model_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath='tb_logs/resnet50/checkpoints/', 
        monitor='evaluation_categorical_accuracy_vs_iterations',
        verbose=0,
        save_best_only=False,
        save_weights_only=False,
        mode="max",
        save_freq="epoch",
    )

    history = model.fit(
        train_data_iterator,
        validation_data=test_data_iterator,
        steps_per_epoch= train_num  // global_batch_size , 
        validation_steps= test_num // global_batch_size ,
        epochs=args.num_train_epochs,
        callbacks=[SavePretrainedCallback(), tensorboard_callback]
    )

