import datasets
from datasets import load_dataset
import datasets
# from transformers import AutoConfig, GPT2LMHeadModel
from transformers import BertTokenizer
import logging

dataset1 = load_dataset('json', data_files={'train': 'datafolder/CLUE/baike_qa.json'})
dataset2 = load_dataset('json', data_files={'train': 'datafolder/CLUE/news2016zh.json'})
dataset3 = load_dataset('json', data_files={'train': 'datafolder/CLUE/webtext2019zh.json'})
dataset4 = load_dataset('json', data_files={'train': 'datafolder/CLUE/translation.json'})
dataset5 = load_dataset('json', data_files={'train': 'datafolder/CLUE/THUCNews.json'})
dataset6 = load_dataset('json', data_files={'train': 'datafolder/CLUE/wiki_zh.json'})

# create text feature from title, desc, and answer for dataset1
# dataset1 = dataset1.map(lambda example: {'text': example['title'] + '。' + example['desc'] + example['answer']}, remove_columns=['qid', 'category', 'title', 'desc','answer'])

# dataset2 = dataset2.map(lambda example: {'text': example['content']}, remove_columns=['news_id', 'keywords', 'desc', 'title', 'source', 'time', 'content'])

# dataset3 = dataset3.map(lambda example: {'text': example['title'] + '。' + example['content']}, remove_columns=['qid', 'title', 'desc', 'topic', 'star', 'content', 'answer_id', 'answerer_tags'])

dataset1 = dataset1.remove_columns(['id'])
dataset2 = dataset2.remove_columns(['id'])
dataset3 = dataset3.remove_columns(['id'])
dataset4 = dataset4.remove_columns(['id'])
dataset5 = dataset5.remove_columns(['id'])
dataset6 = dataset6.remove_columns(['id'])



data = datasets.concatenate_datasets([dataset1['train'],dataset2['train'],dataset3['train'],dataset4['train'], dataset5['train'],dataset6['train']])
print(data)



# tokenizer = BertTokenizer(vocab_file='chinese_L-12_H-768_A-12/vocab.txt')
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
# tokenizer = BertTokenizer.from_pretrained("uer/gpt2-chinese-cluecorpussmall")
max_seq_length = 128

max_seq_length = min(max_seq_length, tokenizer.model_max_length)


def tokenize_function(examples):
    return tokenizer(examples["text"],
                        truncation=True,
                        max_length=max_seq_length, add_special_tokens=False)

logger = logging.getLogger("processing data")




tokenized_datasets = data.map(tokenize_function, batched=True, num_proc=128, remove_columns=["text"], load_from_cache_file=False)

print(tokenized_datasets)


# block_size = tokenizer.model_max_length
block_size = 512
print(block_size)

if block_size > 1024:
    logger.warning(
        f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
        "Picking 1024 instead. You can reduce that value by passing --block_size xxx."
    )
    block_size = 1024

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
    num_proc=128,
    load_from_cache_file=False,
    desc=f"Grouping texts in chunks of {block_size}"
)

print(lm_datasets)
# shuffle the dataset
lm_datasets = datasets.Dataset.shuffle(lm_datasets)
# save dataset
lm_datasets.save_to_disk('./lm_dataset_length{}_chunk{}'.format(max_seq_length, block_size))


