from os import makedirs
from transformers import AutoTokenizer, AutoConfig
import tensorflow as tf
import datasets
from datasets import load_dataset
from transformers import create_optimizer, TFAutoModelForMaskedLM
from functools import partial
import numpy as np
import logging
import math
from sklearn.model_selection import train_test_split
import random

import transformers


class SavePretrainedCallback(tf.keras.callbacks.Callback):
    # Hugging Face models have a save_pretrained() method that saves both the weights and the necessary
    # metadata to allow them to be loaded as a pretrained model in future. This is a simple Keras callback
    # that saves the model with this method after each epoch.
    def __init__(self, output_dir, **kwargs):
        super().__init__()
        self.output_dir = output_dir

    def on_epoch_end(self, epoch, logs={}):
        self.model.save_pretrained(self.output_dir+"/checkpoint_epoch{}".format(epoch))
        
    # def on_step_end(self, savesteps, logs={}):
    #     self.model.save_pretrained(self.output_dir+"/checkpoint_iteration{}".format(savesteps))


# region Data generator
def sample_generator(dataset, tokenizer, mlm_probability=0.15, pad_to_multiple_of=None):
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

logger = logging.getLogger(__name__)

# region Setup logging
# accelerator.is_local_main_process is only True for one process per machine.
logger.setLevel(logging.INFO)
datasets.utils.logging.set_verbosity_warning()
transformers.utils.logging.set_verbosity_info()
# endregion




# data_args
max_seq_length=512 # 1024? 512?
preprocessing_num_workers = 128
overwrite_cache = True
# validation_split_percentage = 0.1
checkpoint = None
config_name = "bert-base-uncased"
tokenizer_name = "bert-base-uncased"
model_name_or_path = "bert-base-uncased"

# #### dataloader ####
bookcorpus = datasets.load_dataset('bookcorpus')
wikipedia = datasets.load_dataset('wikipedia','20200501.en')

wikipedia = wikipedia.remove_columns('title')

print(wikipedia)
print(bookcorpus)

raw_datasets = datasets.concatenate_datasets([bookcorpus['train'],wikipedia['train']])


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
    logger.warning("You are using unknown config.")

if tokenizer_name:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
elif model_name_or_path:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
else:
    raise ValueError(
        "You are instantiating a new tokenizer from scratch. This is not supported by this script."
        "You can do it from another script, save it, and load it from here, using --tokenizer_name."
    )
# endregion


column_names = raw_datasets.column_names
text_column_name = "text" if "text" in column_names else column_names[0]

if max_seq_length is None:
    max_seq_length = tokenizer.model_max_length
    if max_seq_length > 1024:
        logger.warning(
            f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
            "Picking 1024 instead. You can reduce that default value by passing --max_seq_length xxx."
        )
        max_seq_length = 1024
else:
    if max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(max_seq_length, tokenizer.model_max_length)
    
    # should I truncate or keep the remaining?
def tokenize_function(examples):
    return tokenizer(examples[text_column_name], return_special_tokens_mask=True)
    # return tokenizer(examples[text_column_name], return_special_tokens_mask=True, max_length=max_seq_length, truncation=True)
    # return tokenizer(examples[text_column_name])

print(column_names)
# tokenized_datasets = raw_datasets.map(
#     tokenize_function,
#     batched=True,
#     num_proc=preprocessing_num_workers,
#     remove_columns=column_names,
#     load_from_cache_file=not overwrite_cache,
#     desc="Running tokenizer on every text in dataset",
# )


def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= max_seq_length:
        total_length = (total_length // max_seq_length) * max_seq_length
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
        for k, t in concatenated_examples.items()
    }
    return result

# tokenized_datasets = tokenized_datasets.map(
#     group_texts,
#     batched=True,
#     num_proc=preprocessing_num_workers,
#     load_from_cache_file=not overwrite_cache,
#     desc=f"Grouping texts in chunks of {max_seq_length}",
# )

tokenized_datasets = datasets.load_from_disk("./data_bert")
# tokenized_datasets.save_to_disk("./data_bert")

train_dataset = tokenized_datasets

# # should I use this ?
# train_indices, val_indices = train_test_split(
#     list(range(len(train_dataset))), test_size=validation_split_percentage / 100
# )

# eval_dataset = train_dataset.select(val_indices)
# train_dataset = train_dataset.select(train_indices)

# Log a few random samples from the training set:
for index in random.sample(range(len(train_dataset)), 3):
    logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
# endregion


per_device_train_batch_size = 12
# # per_device_eval_batch_size = 4
num_train_epochs = 5
learning_rate = 0.0001
warmup_proportion = 0.1

adam_beta1 = 0.5
adam_beta2 = 0.5
adam_epsilon = 1e-07
weight_decay = 0.9999
output_dir = "./saved_model"


with tf.distribute.MirroredStrategy().scope():

    # region Prepare model
    if checkpoint is not None:
        model = TFAutoModelForMaskedLM.from_pretrained(checkpoint, config=config)
    elif model_name_or_path:
        model = TFAutoModelForMaskedLM.from_pretrained(model_name_or_path, config=config)
    else:
        logger.info("Training new model from scratch")
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
    # eval_generator = partial(sample_generator, eval_dataset, tokenizer)
    # eval_signature = {
    #     feature: tf.TensorSpec(shape=(None,), dtype=tf.int64)
    #     for feature in eval_dataset.features
    #     if feature != "special_tokens_mask"
    # }
    # eval_signature["labels"] = eval_signature["input_ids"]
    # eval_signature = (eval_signature, eval_signature["labels"])
    # tf_eval_dataset = (
    #     tf.data.Dataset.from_generator(eval_generator, output_signature=eval_signature)
    #     .with_options(options)
    #     .batch(batch_size=num_replicas * per_device_eval_batch_size, drop_remainder=True)
    # )
    # endregion



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

    def dummy_loss(y_true, y_pred):
        return tf.reduce_mean(y_pred)

    model.compile(optimizer=optimizer, loss={"loss": dummy_loss})
    # endregion


    # region Training and validation
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {per_device_train_batch_size}")
    logger.info(f"  Total train batch size = {per_device_train_batch_size * num_replicas}")

    history = model.fit(
        tf_train_dataset,
        # validation_data=tf_eval_dataset,
        epochs=int(num_train_epochs),
        steps_per_epoch=len(train_dataset) // (per_device_train_batch_size * num_replicas),
        callbacks=[SavePretrainedCallback(output_dir=output_dir)],
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
