from os import makedirs
from numpy.lib.histograms import histogram
from transformers import AutoTokenizer, AutoConfig
import tensorflow as tf
import datasets
from datasets import load_dataset
from transformers import create_optimizer, TFAutoModelForMaskedLM, AdamWeightDecay
from functools import partial
import numpy as np
import logging
import math
# from sklearn.model_selection import train_test_split
import random
from datetime import datetime
import transformers
import time

# from transformers.utils.dummy_tf_objects import AdamWeightDecay


# class SavePretrainedCallback(tf.keras.callbacks.Callback):
#     # Hugging Face models have a save_pretrained() method that saves both the weights and the necessary
#     # metadata to allow them to be loaded as a pretrained model in future. This is a simple Keras callback
#     # that saves the model with this method after each epoch.
#     def __init__(self, output_dir, **kwargs):
#         super().__init__()
#         self.output_dir = output_dir

#     def on_epoch_end(self, epoch, logs={}):
#         self.model.save_pretrained(self.output_dir+"/checkpoint_epoch{}".format(epoch))

class SavePretrainedCallback(tf.keras.callbacks.Callback):
    # Hugging Face models have a save_pretrained() method that saves both the weights and the necessary
    # metadata to allow them to be loaded as a pretrained model in future. This is a simple Keras callback
    # that saves the model with this method after each epoch.
    def __init__(self, output_dir, **kwargs):
        super().__init__()
        self.output_dir = output_dir
        self.epoch = 0


    def on_epoch_end(self, epoch, logs=None):
        self.epoch += 1
        # self.model.save_pretrained(self.output_dir+"/checkpoint_epoch{}".format(epoch))

    def on_train_batch_end(self, batch, logs=None):
        if batch % 10000 == 0:
            self.model.save_pretrained(self.output_dir+"/checkpoint_epoch{}_iteration{}".format(self.epoch, batch))




# region Data generator
def sample_generator(dataset, tokenizer, mlm_probability=0.15, pad_to_multiple_of=None):
    # time_start = time.time()
    if tokenizer.mask_token is None:
        raise ValueError("This tokenizer does not have a mask token which is necessary for masked language modeling. ")
    # Trim off the last partial batch if present
    sample_ordering = np.random.permutation(len(dataset))

    for sample_idx in sample_ordering:
        example = dataset[int(sample_idx)]
        # Handle dicts with proper padding and conversion to tensor.

        example = tokenizer.pad(example, return_tensors="np", pad_to_multiple_of=pad_to_multiple_of)
        special_tokens_mask = example.pop("special_tokens_mask", None)
        example["input_ids"], example["labels"] = mask_tokens(
            example["input_ids"], mlm_probability, tokenizer, special_tokens_mask=special_tokens_mask
        )
        if tokenizer.pad_token_id is not None:
            example["labels"][example["labels"] == tokenizer.pad_token_id] = -100
        example = {key: tf.convert_to_tensor(arr) for key, arr in example.items()}

        yield example, example["labels"]  # TF needs some kind of labels, even if we don't use them

    return


def mask_tokens(inputs, mlm_probability, tokenizer, special_tokens_mask):
    """
    Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
    """
    labels = np.copy(inputs)
    # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
    probability_matrix = np.random.random_sample(labels.shape)
    special_tokens_mask = special_tokens_mask.astype(np.bool_)

    probability_matrix[special_tokens_mask] = 0.0
    masked_indices = probability_matrix > (1 - mlm_probability)
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = (np.random.random_sample(labels.shape) < 0.8) & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = (np.random.random_sample(labels.shape) < 0.5) & masked_indices & ~indices_replaced
    random_words = np.random.randint(low=0, high=len(tokenizer), size=np.count_nonzero(indices_random), dtype=np.int64)
    inputs[indices_random] = random_words

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels

#### tokenizer ####


logdir = logdir = "logs/roberta_scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, 
                                                    histogram_freq=1,
                                                    profile_batch=0,
                                                    update_freq=10000)






# data_args
max_seq_length=512 # 1024? 512?
preprocessing_num_workers = 128
overwrite_cache = False
# validation_split_percentage = 0.1
# checkpoint = "/home/protago/Xing/pretrainRoberta/roberta_saved_model_ep4over5/checkpoint_epoch1_iteration0"
checkpoint = None
# checkpoint = "bert_saved_model_original"
config_name = "roberta-base"
tokenizer_name = "roberta-base"
model_name_or_path = None


# region Load pretrained model and tokenizer
#
# In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
# download model & vocab.
if checkpoint is not None:
    config = AutoConfig.from_pretrained(checkpoint)
elif config_name:
    config = AutoConfig.from_pretrained(config_name)
elif model_name_or_path:
    config = AutoConfig.from_pretrained(model_name_or_path)
else:
    print.warning("You are using unknown config.")



train_dataset = datasets.load_from_disk("./data_roberta")

print(train_dataset)


per_device_train_batch_size = 16
num_train_epochs = 5
learning_rate = 0.0001 # bs 8k ~ lr 6e-4 500K steps
# learning_rate = 0.05e-4
# learning_rate = 0.01e-4
adam_beta1 = 0.9
adam_beta2 = 0.98
adam_epsilon = 1e-6
weight_decay = 0.01

warmup_proportion = 0.1
# warmup_proportion = 0

output_dir = "./roberta_saved_model_ep5"

tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

with tf.distribute.MirroredStrategy().scope():

    # # region Prepare model
    if checkpoint is not None:
        model = TFAutoModelForMaskedLM.from_pretrained(checkpoint, config=config)
    # elif model_name_or_path:
    #     model = TFAutoModelForMaskedLM.from_pretrained(model_name_or_path, config=config)
    else:
        print("Training new model from scratch")
        model = TFAutoModelForMaskedLM.from_config(config)

    model.resize_token_embeddings(len(tokenizer))
    # endregion

    # region TF Dataset preparation
    num_replicas = tf.distribute.MirroredStrategy().num_replicas_in_sync
    train_generator = partial(sample_generator, train_dataset, tokenizer)
    train_signature = {
        feature: tf.TensorSpec(shape=(None,), dtype=tf.int64)
        for feature in train_dataset.features
        if feature != "special_tokens_mask"
    }
    train_signature["labels"] = train_signature["input_ids"]
    train_signature = (train_signature, train_signature["labels"])
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
    tf_train_dataset = (
        tf.data.Dataset.from_generator(train_generator, output_signature=train_signature)
        .with_options(options)
        .batch(batch_size=num_replicas * per_device_train_batch_size, drop_remainder=True)
        .repeat(int(num_train_epochs))
    )



    # region Optimizer and loss
    batches_per_epoch = len(train_dataset) // (num_replicas * per_device_train_batch_size)
    # Bias and layernorm weights are automatically excluded from the decay
    optimizer, lr_schedule = create_optimizer(
        init_lr=learning_rate,
        num_train_steps=int(num_train_epochs * batches_per_epoch),
        num_warmup_steps=int(warmup_proportion * num_train_epochs * batches_per_epoch),
        adam_beta1=adam_beta1,
        adam_beta2=adam_beta2,
        adam_epsilon=adam_epsilon,
        weight_decay_rate=weight_decay,
    )

    # optimizer, lr_schedule = create_optimizer(
    #     init_lr=learning_rate,
    #     num_train_steps=int(num_train_epochs * batches_per_epoch),
    #     num_warmup_steps=int(warmup_proportion * num_train_epochs * batches_per_epoch),
    #     weight_decay_rate=weight_decay,
    # )

    def dummy_loss(y_true, y_pred):
        return tf.reduce_mean(y_pred)

    model.compile(optimizer=optimizer, loss={"loss": dummy_loss})
    # endregion


    # region Training and validation
    # logger.info("***** Running training *****")
    # logger.info(f"  Num examples = {len(train_dataset)}")
    # logger.info(f"  Num Epochs = {num_train_epochs}")
    # logger.info(f"  Instantaneous batch size per device = {per_device_train_batch_size}")
    # logger.info(f"  Total train batch size = {per_device_train_batch_size * num_replicas}")

    history = model.fit(
        tf_train_dataset,
        # validation_data=tf_eval_dataset,
        epochs=int(num_train_epochs),
        steps_per_epoch=len(train_dataset) // (per_device_train_batch_size * num_replicas),
        callbacks=[SavePretrainedCallback(output_dir=output_dir),tensorboard_callback],
    )
    # try:
    #     train_perplexity = math.exp(history.history["loss"][-1])
    # except OverflowError:
    #     train_perplexity = math.inf
    # try:
    #     validation_perplexity = math.exp(history.history["val_loss"][-1])
    # except OverflowError:
    #     validation_perplexity = math.inf
    # logger.warning(f"  Final train loss: {history.history['loss'][-1]:.3f}")
    # logger.warning(f"  Final train perplexity: {train_perplexity:.3f}")
    # logger.warning(f"  Final validation loss: {history.history['val_loss'][-1]:.3f}")
    # logger.warning(f"  Final validation perplexity: {validation_perplexity:.3f}")
    # endregion

    if output_dir is not None:
        model.save_pretrained(output_dir)
