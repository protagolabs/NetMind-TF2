import sys
import os
import json
import datasets
import numpy as np
import logging
import config as c
import tensorflow as tf
from datetime import datetime
from functools import partial
from transformers import AutoTokenizer, AutoConfig
from transformers import create_optimizer, TFAutoModelForMaskedLM, AdamWeightDecay
from NetmindMixins.Netmind import nmp, NetmindDistributedModel, NetmindOptimizer, NetmindDistributedModel
from arguments import setup_args

args = setup_args()

logger = logging.getLogger()
formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(filename)s:%(lineno)d %(message)s', '%Y-%m-%d %H:%M:%S')
console_handler = logging.StreamHandler(sys.stdout)
console_handler.formatter = formatter
logger.addHandler(console_handler)
logger.setLevel(logging.INFO)
# from transformers.utils.dummy_tf_objects import AdamWeightDecay

class SavePretrainedCallback(tf.keras.callbacks.Callback):
    # Hugging Face models have a save_pretrained() method that saves both the weights and the necessary
    # metadata to allow them to be loaded as a pretrained model in future. This is a simple Keras callback
    # that saves the model with this method after each epoch.
    def __init__(self, output_dir, **kwargs):
        super().__init__()
        self.output_dir = output_dir
        self.epoch = 0

    def on_train_begin(self, logs=None):
        logger.info(f'on_train_begin : log : {logs}')
        # here we init train&eval bar
        nmp.init(load_checkpoint=False)
        NetmindDistributedModel(self.model)
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
        logger.info(f'on_train_batch_end : batch : {batch} , log : {logs}')

        learning_rate = self.model.optimizer.learning_rate(self.model.optimizer.iterations.numpy())

        learning_rate = tf.keras.backend.get_value(learning_rate)
        logger.info(f'learning_rate : {learning_rate}')

        nmp.step({"loss": float(logs['loss']),
                  "Learning rate": float(learning_rate)})
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

#### tokenizer ####


if __name__ == '__main__':

    if not os.getenv('TF_CONFIG'):
        c.tf_config['task']['index'] = int(os.getenv('INDEX'))
        os.environ['TF_CONFIG'] = json.dumps(c.tf_config)

    n_workers = len(json.loads(os.environ['TF_CONFIG']).get('cluster', {}).get('worker'))
    logger.info(f'c.tf_config : {c.tf_config}')

    logdir = logdir = "logs/roberta_scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir,
                                                        histogram_freq=1,
                                                        profile_batch=0,
                                                        update_freq=10000)




    checkpoint = None
    model_name_or_path = None
    """
    # data_args
    args.max_seq_length=512 # 1024? 512?
    args.preprocessing_num_workers = 128
    args.overwrite_cache = False
    # validation_split_percentage = 0.1
    # checkpoint = "/home/protago/Xing/pretrainRoberta/roberta_saved_model_ep4over5/checkpoint_epoch1_iteration0"
    # checkpoint = "bert_saved_model_original"
    args.config_name = "roberta-base"
    args.tokenizer_name = "roberta-base"
    """


    # region Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if checkpoint is not None:
        config = AutoConfig.from_pretrained(checkpoint)
    elif args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif model_name_or_path:
        config = AutoConfig.from_pretrained(model_name_or_path)
    else:
        print.warning("You are using unknown config.")



    train_dataset = datasets.load_from_disk("./data_roberta")

    print(train_dataset)

    """
    args.per_device_train_batch_size = 16
    args.num_train_epochs = 5
    args.learning_rate = 0.0001 # bs 8k ~ lr 6e-4 500K steps
    # args.learning_rate = 0.05e-4
    # args.learning_rate = 0.01e-4
    args.adam_beta1 = 0.9
    args.adam_beta2 = 0.98
    args.adam_epsilon = 1e-6
    args.weight_decay = 0.01
    
    args.warmup_proportion = 0.1
    # args.warmup_proportion = 0
    
    args.output_dir = "./roberta_saved_model_ep5"
    """

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    with tf.distribute.MultiWorkerMirroredStrategy().scope():

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
        num_replicas = tf.distribute.MultiWorkerMirroredStrategy().num_replicas_in_sync
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
            .batch(batch_size=num_replicas * args.per_device_train_batch_size, drop_remainder=True)
            .repeat(int(args.num_train_epochs))
        )



        # region Optimizer and loss
        batches_per_epoch = len(train_dataset) // (num_replicas * args.per_device_train_batch_size)
        # Bias and layernorm weights are automatically excluded from the decay

        optimizer, lr_schedule = create_optimizer(
            init_lr=args.learning_rate,
            num_train_steps=int(args.num_train_epochs * batches_per_epoch),
            num_warmup_steps=int(args.warmup_proportion * args.num_train_epochs * batches_per_epoch),
            adam_beta1=args.adam_beta1,
            adam_beta2=args.adam_beta2,
            adam_epsilon=args.adam_epsilon,
            weight_decay_rate=args.weight_decay,
        )


        # optimizer, lr_schedule = create_optimizer(
        #     init_lr=args.learning_rate,
        #     num_train_steps=int(args.num_train_epochs * batches_per_epoch),
        #     num_warmup_steps=int(args.warmup_proportion * args.num_train_epochs * batches_per_epoch),
        #     args.weight_decay_rate=args.weight_decay,
        # )

        def dummy_loss(y_true, y_pred):
            return tf.reduce_mean(y_pred)

        model.compile(optimizer=optimizer, loss={"loss": dummy_loss})
        # endregion


        history = model.fit(
            tf_train_dataset,
            # validation_data=tf_eval_dataset,
            epochs=int(args.num_train_epochs),
            steps_per_epoch=len(train_dataset) // (args.per_device_train_batch_size * num_replicas),
            callbacks=[SavePretrainedCallback(output_dir=args.output_dir),tensorboard_callback],
        )

