import sys
import json
import tensorflow as tf
import random
import logging
import os
import datasets
import numpy as np
import config as c
# from datasets import load_dataset
from transformers import create_optimizer, TFAutoModelForMaskedLM
from functools import partial
from datetime import datetime
from transformers import AutoTokenizer, AutoConfig
import transformers
from NetmindMixins.Netmind import nmp, NetmindDistributedModel, NetmindOptimizer, NetmindDistributedModel

logger = logging.getLogger()
formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(filename)s:%(lineno)d %(message)s', '%Y-%m-%d %H:%M:%S')

console_handler = logging.StreamHandler(sys.stdout)
console_handler.formatter = formatter
logger.addHandler(console_handler)
logger.setLevel(logging.INFO)

from arguments import setup_args

args = setup_args()


class SavePretrainedCallback(tf.keras.callbacks.Callback):
    # Hugging Face models have a save_pretrained() method that saves both the weights and the necessary
    # metadata to allow them to be loaded as a pretrained model in future. This is a simple Keras callback
    # that saves the model with this method after each epoch.
    def __init__(self, output_dir, **kwargs):
        super().__init__()
        self.output_dir = output_dir
        self.epoch = 0

    def on_train_begin(self, logs=None):
        logger.info(f'log : {logs}')
        # here we init train&eval bar
        nmp.init_train_bar(total_epoch=args.num_train_epochs, step_per_epoch=batches_per_epoch)
        nmp.init_eval_bar(total_epoch=args.num_train_epochs)
        epochs_trained = nmp.cur_epoch
        logger.info(f'epochs_trained: {epochs_trained}')

    def on_train_end(self, logs=None):
        logger.info(f'log : {logs}')
        nmp.finish_training()

    def on_train_batch_begin(self, batch, logs=None):
        logger.info(f'batch : {batch}, logs: {logs}')
        if nmp.should_skip_step():
            return

    def on_train_batch_end(self, batch, logs=None):
        logger.info(f'batch : {batch} , log : {logs}')
        nmp.step({"loss": float(logs['loss']),
                  "Learning rate": float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))})
        logger.info(f'save_pretrained_by_step : {args.save_steps}')
        nmp.save_pretrained_by_step(args.save_steps)

    def on_test_end(self, logs=None):
        logger.info(f'log : {logs}')
        nmp.evaluate(logs)

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


if __name__ == '__main__':
    from tensorflow.python.client import device_lib

    logger.info(device_lib.list_local_devices())
    if not os.getenv('TF_CONFIG'):
        c.tf_config['task']['index'] = int(os.getenv('INDEX'))
        os.environ['TF_CONFIG'] = json.dumps(c.tf_config)

    n_workers = len(json.loads(os.environ['TF_CONFIG']).get('cluster', {}).get('worker'))
    global_batch_size = args.per_device_train_batch_size * n_workers

    logdir = "logs/bert_scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir,
                                                          histogram_freq=1,
                                                          profile_batch=0,
                                                          update_freq=10000)

    """
    per_device_train_batch_size = 16
    num_train_epochs = 6
    learning_rate = 0.0001
    warmup_proportion = 0.1

    weight_decay = 1e-7
    output_dir = "./bert_saved_model_ep6"

    # data_args
    max_seq_length=512 # 1024? 512?
    preprocessing_num_workers = 128
    overwrite_cache = True

    config_name = "bert-base-uncased"
    tokenizer_name = "bert-base-uncased"
    model_name_or_path = None # for training from scratch
    """
    checkpoint = None

    # # #### dataloader ####
    # bookcorpus = datasets.load_dataset('bookcorpus')
    # # wikipedia = datasets.load_dataset('wikipedia','20200501.en')
    # wikipedia = datasets.load_dataset('wikipedia','20220301.en')
    # wikipedia = wikipedia.remove_columns('title')

    # print(wikipedia)
    # print(bookcorpus)

    # # raw_datasets = datasets.concatenate_datasets([bookcorpus['train'],wikipedia['train']])
    # raw_datasets = bookcorpus['train']
    # print(raw_datasets)

    # region Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if checkpoint is not None:
        config = AutoConfig.from_pretrained(checkpoint)
    elif args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        print("You are using unknown config.")

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --args.tokenizer_name."
        )
    # endregion

    # column_names = raw_datasets.column_names
    # text_column_name = "text" if "text" in column_names else column_names[0]
    # print(text_column_name)
    if args.max_seq_length is None:
        args.max_seq_length = tokenizer.model_max_length
        if args.max_seq_length > 1024:
            print(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can reduce that default value by passing --max_seq_length xxx."
            )
            args.max_seq_length = 1024
    else:
        if args.max_seq_length > tokenizer.model_max_length:
            print(
                f"The args.max_seq_length passed ({args.max_seq_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using args.max_seq_length={tokenizer.model_max_length}."
            )
        args.max_seq_length = min(args.max_seq_length, tokenizer.model_max_length)

        # should I truncate or keep the remaining?
    # def tokenize_function(examples):
    #     return tokenizer(examples[text_column_name], return_special_tokens_mask=True)
    # return tokenizer(examples[text_column_name], return_special_tokens_mask=True, max_length=args.max_seq_length, truncation=True)
    # return tokenizer(examples[text_column_name])

    # tokenized_datasets = raw_datasets.map(
    #     tokenize_function,
    #     batched=True,
    #     num_proc=args.preprocessing_num_workers,
    #     remove_columns=column_names,
    #     load_from_cache_file=not args.overwrite_cache,
    #     desc="Running tokenizer on every text in dataset",
    # )

    # def group_texts(examples):
    #     # Concatenate all texts.
    #     concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    #     total_length = len(concatenated_examples[list(examples.keys())[0]])
    #     # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    #     # customize this part to your needs.
    #     if total_length >= args.max_seq_length:
    #         total_length = (total_length // args.max_seq_length) * args.max_seq_length
    #     # Split by chunks of max_len.
    #     result = {
    #         k: [t[i : i + args.max_seq_length] for i in range(0, total_length, args.max_seq_length)]
    #         for k, t in concatenated_examples.items()
    #     }
    #     return result

    # tokenized_datasets = tokenized_datasets.map(
    #     group_texts,
    #     batched=True,
    #     num_proc=args.preprocessing_num_workers,
    #     load_from_cache_file=not args.overwrite_cache,
    #     desc=f"Grouping texts in chunks of {args.max_seq_length}",
    # )

    #### you can save/load the preprocessed data here ###

    tokenized_datasets = datasets.load_from_disk("./data_bert")
    # tokenized_datasets.save_to_disk("./data_bert")

    train_dataset = tokenized_datasets
    print(train_dataset)

    # train_indices, val_indices = train_test_split(
    #     list(range(len(train_dataset))), test_size=validation_split_percentage / 100
    # )

    # eval_dataset = train_dataset.select(val_indices)
    # train_dataset = train_dataset.select(train_indices)

    # Log a few random samples from the training set:
    # for index in random.sample(range(len(train_dataset)), 3):
    #     print(f"Sample {index} of the training set: {train_dataset[index]}.")
    # endregion

    batches_per_epoch = len(train_dataset) // (n_workers * args.per_device_train_batch_size)

    with tf.distribute.MultiWorkerMirroredStrategy().scope():

        # # region Prepare model
        if checkpoint is not None:
            model = TFAutoModelForMaskedLM.from_pretrained(checkpoint, config=config)
        elif args.model_name_or_path:
            model = TFAutoModelForMaskedLM.from_pretrained(args.model_name_or_path, config=config)
        else:
            print("Training new model from scratch")
            model = TFAutoModelForMaskedLM.from_config(config)

        model.resize_token_embeddings(len(tokenizer))
        # endregion

        # region TF Dataset preparation
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
                .batch(batch_size=n_workers * args.per_device_train_batch_size, drop_remainder=True)
                .repeat(int(args.num_train_epochs))
        )
        # endregion

        # region Optimizer and loss

        # Bias and layernorm weights are automatically excluded from the decay
        # optimizer, lr_schedule = create_optimizer(
        #     init_lr=learning_rate,
        #     num_train_steps=int(num_train_epochs * batches_per_epoch),
        #     num_warmup_steps=int(warmup_proportion * num_train_epochs * batches_per_epoch),
        #     adam_beta1=adam_beta1,
        #     adam_beta2=adam_beta2,
        #     adam_epsilon=adam_epsilon,
        #     weight_decay_rate=weight_decay,
        # )

        optimizer, lr_schedule = create_optimizer(
            init_lr=args.learning_rate,
            num_train_steps=int(args.num_train_epochs * batches_per_epoch),
            num_warmup_steps=int(args.warmup_proportion * args.num_train_epochs * batches_per_epoch),
            weight_decay_rate=args.weight_decay,
        )


        def dummy_loss(y_true, y_pred):
            return tf.reduce_mean(y_pred)


        model.compile(optimizer=optimizer, loss={"loss": dummy_loss})
        # endregion

        # region Training

        history = model.fit(
            tf_train_dataset,
            # validation_data=tf_eval_dataset,
            epochs=int(args.num_train_epochs),
            steps_per_epoch=len(train_dataset) // (args.per_device_train_batch_size * n_workers),
            callbacks=[SavePretrainedCallback(output_dir=args.output_dir), tensorboard_callback],
        )

        if args.output_dir is not None:
            model.save_pretrained(args.output_dir)
