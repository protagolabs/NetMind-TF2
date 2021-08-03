from os import makedirs
from transformers import BertTokenizerFast
# from transformers import TextDataset,DataCollatorForLanguageModeling
import tensorflow as tf
import datasets
from datasets import load_dataset
from transformers import create_optimizer, TFAutoModelForCausalLM
from functools import partial
import numpy as np
import logging
import math

# region Data generator
def sample_generator(dataset, tokenizer):
    # Trim off the last partial batch if present
    sample_ordering = np.random.permutation(len(dataset))
    for sample_idx in sample_ordering:
        example = dataset[int(sample_idx)]
        # Handle dicts with proper padding and conversion to tensor.
        example = {key: tf.convert_to_tensor(arr, dtype_hint=tf.int64) for key, arr in example.items()}
        yield example, example["labels"]  # TF needs some kind of labels, even if we don't use them
    return

class SavePretrainedCallback(tf.keras.callbacks.Callback):
    # Hugging Face models have a save_pretrained() method that saves both the weights and the necessary
    # metadata to allow them to be loaded as a pretrained model in future. This is a simple Keras callback
    # that saves the model with this method after each epoch.
    def __init__(self, output_dir, **kwargs):
        super().__init__()
        self.output_dir = output_dir

    def on_epoch_end(self, epoch, logs=None):
        
        self.model.save_pretrained(self.output_dir+"/checkpoint_epoch{}".format(epoch))

#### tokenizer ####

logger = logging.getLogger(__name__)

tokenizer = BertTokenizerFast.from_pretrained('uer/gpt2-chinese-cluecorpussmall')


max_seq_length=128
preprocessing_num_workers = 2
overwrite_cache = True

if max_seq_length > tokenizer.model_max_length:
    print(f"The max_seq_length passed ({max_seq_length}) is larger than the maximum length for the"
        f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
    )
#### dataloader ####
train_path = 'train_dataset.txt'
test_path = 'test_dataset.txt'

raw_datasets = load_dataset('text', data_files={'train':train_path, 'validation':test_path})

column_names = raw_datasets["train"].column_names
text_column_name = "text" if "text" in column_names else column_names[0]

def tokenize_function(examples):
    return tokenizer(examples[text_column_name])


tokenized_datasets = raw_datasets.map(
    tokenize_function,
    batched=True,
    num_proc=preprocessing_num_workers,
    remove_columns=column_names,
    load_from_cache_file=not overwrite_cache,
    desc="Running tokenizer on dataset",
)

# block_size = tokenizer.model_max_length
block_size = 256

if block_size > 1024:
    print(
        f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
        "Picking 1024 instead. You can reduce that value by passing --block_size xxx."
    )
    block_size = 1024

# Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    num_proc=preprocessing_num_workers,
    load_from_cache_file=not overwrite_cache,
    desc=f"Grouping texts in chunks of {block_size}",
)

train_dataset = lm_datasets["train"]
eval_dataset = lm_datasets["validation"]

per_device_train_batch_size = 4
per_device_eval_batch_size = 4
num_train_epochs = 3
learning_rate = 0.0001
warmup_steps = 1000

adam_beta1 = 0.5
adam_beta2 = 0.5
adam_epsilon = 1e-07
weight_decay = 0.9999
output_dir = "./gpt2-chinese-sm-models-tf2"


with tf.distribute.MirroredStrategy().scope():

    model = TFAutoModelForCausalLM.from_pretrained("uer/gpt2-chinese-cluecorpussmall")
    model.resize_token_embeddings(len(tokenizer))


    num_replicas = tf.distribute.MirroredStrategy().num_replicas_in_sync
    train_generator = partial(sample_generator, train_dataset, tokenizer)
    train_signature = {
        feature: tf.TensorSpec(shape=(None,), dtype=tf.int64)
        for feature in train_dataset.features
        if feature != "special_tokens_mask"
    }
    train_sig = (train_signature, train_signature["labels"])
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
    tf_train_dataset = (
        tf.data.Dataset.from_generator(train_generator, output_signature=train_sig)
        .with_options(options)
        .batch(batch_size=num_replicas * per_device_train_batch_size, drop_remainder=True)
        .repeat(int(num_train_epochs))
    )
    eval_generator = partial(sample_generator, eval_dataset, tokenizer)
    eval_signature = {
        feature: tf.TensorSpec(shape=(None,), dtype=tf.int64)
        for feature in eval_dataset.features
        if feature != "special_tokens_mask"
    }
    eval_sig = (eval_signature, eval_signature["labels"])
    tf_eval_dataset = (
        tf.data.Dataset.from_generator(eval_generator, output_signature=eval_sig)
        .with_options(options)
        .batch(batch_size=num_replicas * per_device_eval_batch_size, drop_remainder=True)
        .repeat(int(num_train_epochs))
    )


    # region Optimizer and loss
    batches_per_epoch = len(train_dataset) // (num_replicas * per_device_train_batch_size)
    # Bias and layernorm weights are automatically excluded from the decay
    optimizer, lr_schedule = create_optimizer(
        init_lr=learning_rate,
        num_train_steps=int(num_train_epochs * batches_per_epoch),
        num_warmup_steps=warmup_steps,
        adam_beta1=adam_beta1,
        adam_beta2=adam_beta2,
        adam_epsilon=adam_epsilon,
        weight_decay_rate=weight_decay,
    )

    def dummy_loss(y_true, y_pred):
        return tf.reduce_mean(y_pred)

    model.compile(optimizer=optimizer, loss={"loss": dummy_loss})

    history = model.fit(
        tf_train_dataset,
        validation_data=tf_eval_dataset,
        epochs=int(num_train_epochs),
        steps_per_epoch=len(train_dataset) // (per_device_train_batch_size * num_replicas),
        callbacks=[SavePretrainedCallback(output_dir=output_dir)],
    )

    try:
        train_perplexity = math.exp(history.history["loss"][-1])
    except OverflowError:
        train_perplexity = math.inf
    try:
        validation_perplexity = math.exp(history.history["val_loss"][-1])
    except OverflowError:
        validation_perplexity = math.inf

    logger.info(f"  Final train loss: {history.history['loss'][-1]:.3f}")
    logger.info(f"  Final train perplexity: {train_perplexity:.3f}")
    logger.info(f"  Final validation loss: {history.history['val_loss'][-1]:.3f}")
    logger.info(f"  Final validation perplexity: {validation_perplexity:.3f}")
    
    if output_dir is not None:
        model.save_pretrained(output_dir)
