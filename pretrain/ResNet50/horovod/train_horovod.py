#
# ResNet-50 model training using Keras and Horovod.
#
# This model is an example of a computation-intensive model that achieves good accuracy on an image
# classification task.  It brings together distributed training concepts such as learning rate
# schedule adjustments with a warmup, randomized data reading, and checkpointing on the first worker
# only.
#
# Note: This model uses Keras native ImageDataGenerator and not the sophisticated preprocessing
# pipeline that is typically used to train state-of-the-art ResNet-50 model.  This results in ~0.5%
# increase in the top-1 validation error compared to the single-crop top-1 validation error from
# https://github.com/KaimingHe/deep-residual-networks.
#
import argparse
import keras
from keras import backend as K
from keras.preprocessing import image
import tensorflow as tf
from timeit import default_timer as timer
import horovod.keras as hvd
import os
import tensorflow as tf
import numpy as np
from arguments import setup_args
from NetmindMixins.Netmind import nmp, NetmindDistributedModel, NetmindOptimizer
from NetmindMixins.Netmind import  HorovodCallback

device = "GPU"

def get_train_data(args):
    # Training data iterator.
    train_gen = image.ImageDataGenerator(
        width_shift_range=0.33, height_shift_range=0.33, zoom_range=0.5, horizontal_flip=True,
        preprocessing_function=keras.applications.resnet50.preprocess_input)
    train_iter = train_gen.flow_from_directory(args.train_dir,
                                               batch_size=args.per_device_train_batch_size,
                                               target_size=(224, 224))
    return train_iter

def get_test_data(args):
    # Validation data iterator.
    test_gen = image.ImageDataGenerator(
        zoom_range=(0.875, 0.875), preprocessing_function=keras.applications.resnet50.preprocess_input)
    test_iter = test_gen.flow_from_directory(args.val_dir,
                                             batch_size=args.val_batch_size,
                                             target_size=(224, 224))
    return test_iter

def get_model():
    # Set up standard ResNet-50 model.
    model = keras.applications.resnet50.ResNet50(weights=None)
    return model

class TimingCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.img_secs = []

    def on_train_end(self, logs=None):
        img_sec_mean = np.mean(self.img_secs)
        img_sec_conf = 1.96 * np.std(self.img_secs)
        print('Img/sec per %s: %.1f +-%.1f' % (device, img_sec_mean, img_sec_conf))
        print('Total img/sec on %d %s(s): %.1f +-%.1f' %
             (hvd.size(), device, hvd.size() * img_sec_mean, hvd.size() * img_sec_conf))

    def on_epoch_begin(self, epoch, logs=None):
        self.starttime = timer()

    def on_epoch_end(self, epoch, logs=None):
        time = timer() - self.starttime
        img_sec = args.batch_size * args.num_batches_per_iter / time
        print('Iter #%d: %.1f img/sec per %s' % (epoch, img_sec, device))
        # skip warm up epoch
        if epoch > 0:
            self.img_secs.append(img_sec)

def main(args):

    hvd.init()

    # Horovod: pin GPU to be used to process local rank (one GPU per process)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(hvd.local_rank())
    K.set_session(tf.Session(config=config))

    # If set > 0, will resume training from a given checkpoint.
    resume_from_epoch = 0
    for try_epoch in range(args.per_device_train_batch_size, 0, -1):
        if os.path.exists(args.checkpoint_format.format(epoch=try_epoch)):
            resume_from_epoch = try_epoch
            break

    # Horovod: broadcast resume_from_epoch from rank 0 (which will have
    # checkpoints) to other ranks.
    resume_from_epoch = hvd.broadcast(resume_from_epoch, 0, name='resume_from_epoch')

    # Horovod: print logs on the first worker.
    verbose = 1 if hvd.rank() == 0 else 0

    #get train dataset
    train_iter = get_train_data(args)

    test_iter = None
    if args.val_dir and args.val_batch_size:
        test_iter = get_test_data(args)

    model = get_model()

    # Horovod: (optional) compression algorithm.
    compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none

    # Horovod: adjust learning rate based on number of GPUs.
    initial_lr = args.learning_rate * hvd.size()

    # Restore from a previous checkpoint, if initial_epoch is specified.
    # Horovod: restore on the first worker which will broadcast both model and optimizer weights
    # to other workers.
    if resume_from_epoch > 0 and hvd.rank() == 0:
        model = hvd.load_model(args.checkpoint_format.format(epoch=resume_from_epoch),
                               compression=compression)
    else:
        # ResNet-50 model that is included with Keras is optimized for inference.
        # Add L2 weight decay & adjust BN settings.
        model_config = model.get_config()
        for layer, layer_config in zip(model.layers, model_config['layers']):
            if hasattr(layer, 'kernel_regularizer'):
                regularizer = keras.regularizers.l2(args.wd)
                layer_config['config']['kernel_regularizer'] = \
                    {'class_name': regularizer.__class__.__name__,
                     'config': regularizer.get_config()}
            if type(layer) == keras.layers.BatchNormalization:
                layer_config['config']['momentum'] = 0.9
                layer_config['config']['epsilon'] = 1e-5

        model = keras.models.Model.from_config(model_config)
        opt = keras.optimizers.SGD(lr=initial_lr, momentum=args.momentum)

        # Horovod: add Horovod Distributed Optimizer.
        opt = hvd.DistributedOptimizer(opt, compression=compression)

        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=opt,
                      metrics=['accuracy', 'top_k_categorical_accuracy'])

    callbacks = [
        # Horovod: broadcast initial variable states from rank 0 to all other processes.
        # This is necessary to ensure consistent initialization of all workers when
        # training is started with random weights or restored from a checkpoint.
        hvd.callbacks.BroadcastGlobalVariablesCallback(0),

        # Horovod: average metrics among workers at the end of every epoch.
        #
        # Note: This callback must be in the list before the ReduceLROnPlateau,
        # TensorBoard, or other metrics-based callbacks.
        hvd.callbacks.MetricAverageCallback(),

        # Horovod: using `lr = 1.0 * hvd.size()` from the very beginning leads to worse final
        # accuracy. Scale the learning rate `lr = 1.0` ---> `lr = 1.0 * hvd.size()` during
        # the first five epochs. See https://arxiv.org/abs/1706.02677 for details.
        hvd.callbacks.LearningRateWarmupCallback(initial_lr=initial_lr,
                                                 warmup_epochs=args.warmup_epochs,
                                                 verbose=verbose),

        # Horovod: after the warmup reduce learning rate by 10 on the 30th, 60th and 80th epochs.
        hvd.callbacks.LearningRateScheduleCallback(initial_lr=initial_lr,
                                                   multiplier=1.,
                                                   start_epoch=args.warmup_epochs,
                                                   end_epoch=30),
        hvd.callbacks.LearningRateScheduleCallback(initial_lr=initial_lr, multiplier=1e-1, start_epoch=30,
                                                   end_epoch=60),
        hvd.callbacks.LearningRateScheduleCallback(initial_lr=initial_lr, multiplier=1e-2, start_epoch=60,
                                                   end_epoch=80),
        hvd.callbacks.LearningRateScheduleCallback(initial_lr=initial_lr, multiplier=1e-3, start_epoch=80),
    ]

    # Horovod: save checkpoints only on the first worker to prevent other workers from corrupting them.
    if hvd.rank() == 0:
        callbacks.append(keras.callbacks.ModelCheckpoint(args.checkpoint_format))
        callbacks.append(keras.callbacks.TensorBoard(args.log_dir))
        callbacks.append(HorovodCallback)

    # Train the model. The training will randomly sample 1 / N batches of training data and
    # 3 / N batches of validation data on every worker, where N is the number of workers.
    # Over-sampling of validation data helps to increase probability that every validation
    # example will be evaluated.
    model.fit_generator(train_iter,
                        steps_per_epoch=len(train_iter) // hvd.size(),
                        callbacks=callbacks,
                        epochs=args.num_train_epochs,
                        verbose=verbose,
                        workers=4,
                        initial_epoch=resume_from_epoch,
                        validation_data=test_iter,
                        validation_steps=3 * len(test_iter) // hvd.size())

    # Evaluate the model on the full data set.
    score = hvd.allreduce(model.evaluate_generator(test_iter, len(test_iter), workers=4))
    if verbose:
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

if __name__ == '__main__':
    args = setup_args()
    main(args)
    nmp.finish_training()