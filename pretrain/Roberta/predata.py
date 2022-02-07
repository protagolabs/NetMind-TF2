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





# data_args
max_seq_length=512 # 1024? 512?
preprocessing_num_workers = 32
overwrite_cache = True
tokenizer_name = "roberta-base"


# #### dataloader ####
# #### dataloader ####
bookcorpus = datasets.load_dataset('bookcorpus')
print(bookcorpus)
wikipedia = datasets.load_dataset('wikipedia','20200501.en')
print(wikipedia)
ccnews = datasets.load_dataset('cc_news')
print(ccnews)
openwebtxt = datasets.load_dataset('openwebtext')
print(openwebtxt)

wikipedia = wikipedia.remove_columns('title')
ccnews = ccnews.remove_columns(['title', 'domain', 'date', 'description', 'url', 'image_url'])

raw_datasets = datasets.concatenate_datasets([bookcorpus['train'],wikipedia['train'],ccnews['train'],openwebtxt['train']])
print(raw_datasets)



if tokenizer_name:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
else:
    raise ValueError(
        "You are instantiating a new tokenizer from scratch. This is not supported by this script."
        "You can do it from another script, save it, and load it from here, using --tokenizer_name."
    )
# endregion


column_names = raw_datasets.column_names
text_column_name = "text" if "text" in column_names else column_names[0]
print(text_column_name)
if max_seq_length is None:
    max_seq_length = tokenizer.model_max_length
    if max_seq_length > 1024:
        print.warning(
            f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
            "Picking 1024 instead. You can reduce that default value by passing --max_seq_length xxx."
        )
        max_seq_length = 1024
else:
    if max_seq_length > tokenizer.model_max_length:
        print.warning(
            f"The max_seq_length passed ({max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(max_seq_length, tokenizer.model_max_length)
    
    # should I truncate or keep the remaining?
def tokenize_function(examples):
    return tokenizer(examples[text_column_name], return_special_tokens_mask=True)
    # return tokenizer(examples[text_column_name], return_special_tokens_mask=True, max_length=max_seq_length, truncation=True)
    # return tokenizer(examples[text_column_name])


tokenized_datasets = raw_datasets.map(
    tokenize_function,
    batched=True,
    num_proc=preprocessing_num_workers,
    remove_columns=column_names,
    load_from_cache_file=not overwrite_cache,
    desc="Running tokenizer on every text in dataset",
)


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

tokenized_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    num_proc=preprocessing_num_workers,
    load_from_cache_file=not overwrite_cache,
    desc=f"Grouping texts in chunks of {max_seq_length}",
)

tokenized_datasets.save_to_disk("./data_roberta")


