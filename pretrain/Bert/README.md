# Bert
In this repo we pretrain [Bert](https://arxiv.org/abs/1810.04805) by using the huggingface transofmers with Tensorflow 2.

We use [wikipedia](https://huggingface.co/datasets/wikipedia) and [bookcorpus](https://huggingface.co/datasets/bookcorpus) for training this model and evaluate the results on [GLUE](https://gluebenchmark.com/).

## training ##

python **train.py**

## evaluation ##

please run the bash files in the directory **glue_eval** 

## Result 

| GLUE | CoLA | SST-2 | MRPC | STS-B | QQP | MNLI | QNLI | RTE | WNLI |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |



## Acknowledgement ##
[Huggingface Transformers Language Modeling](https://github.com/huggingface/transformers/blob/master/examples/tensorflow/language-modeling/run_mlm.py)

Thanks for my colleages Xiangpeng Wan and Yu Cheng for their kindly helps
