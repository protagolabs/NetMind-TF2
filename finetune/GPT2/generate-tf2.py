from transformers import pipeline
from transformers import BertTokenizerFast

tokenizer = BertTokenizerFast.from_pretrained('uer/gpt2-chinese-cluecorpussmall')

chef = pipeline('text-generation',model='./gpt2-chinese-sm-models-tf2/', tokenizer=tokenizer)

print(chef('[CLS]你会谅解我吗[SEP]', max_length=128,  pad_token_id=102, do_sample=True, top_k=5, eos_token_id=tokenizer.get_vocab().get("[SEP]", 0))[0]['generated_text'])
print(chef('[CLS]我过得非常寂寞[SEP]', max_length=128, pad_token_id=102, do_sample=True, top_k=5, eos_token_id=tokenizer.get_vocab().get("[SEP]", 0))[0]['generated_text'])
print(chef('[CLS]说真的你觉得我怎么样[SEP]', max_length=128, pad_token_id=102, do_sample=True, top_k=5, eos_token_id=tokenizer.get_vocab().get("[SEP]", 0))[0]['generated_text'])
print(chef('[CLS]我觉得你和天猫精灵比差远了[SEP]', max_length=128, pad_token_id=102, do_sample=True, top_k=5, eos_token_id=tokenizer.get_vocab().get("[SEP]", 0))[0]['generated_text'])
print(chef('[CLS]我感觉有点饿[SEP]', max_length=128, pad_token_id=102, do_sample=True, top_k=5, eos_token_id=tokenizer.get_vocab().get("[SEP]", 0))[0]['generated_text'])
print(chef('[CLS]你现在有在相亲吗[SEP]', max_length=128,  pad_token_id=102, do_sample=True, top_k=5, eos_token_id=tokenizer.get_vocab().get("[SEP]", 0))[0]['generated_text'])
print(chef('[CLS]和你聊天真的是好没趣[SEP]', max_length=128, pad_token_id=102, do_sample=True, top_k=5, eos_token_id=tokenizer.get_vocab().get("[SEP]", 0))[0]['generated_text'])
print(chef('[CLS]谁负责管理你的[SEP]', max_length=128, pad_token_id=102, do_sample=True, top_k=5, eos_token_id=tokenizer.get_vocab().get("[SEP]", 0))[0]['generated_text'])
print(chef('[CLS]你在跟我开玩笑吗[SEP]', max_length=128, pad_token_id=102, do_sample=True, top_k=5, eos_token_id=tokenizer.get_vocab().get("[SEP]", 0))[0]['generated_text'])
print(chef('[CLS]我很想成为你的永远的好朋友[SEP]', max_length=128, pad_token_id=102, do_sample=True, top_k=5, eos_token_id=tokenizer.get_vocab().get("[SEP]", 0))[0]['generated_text'])