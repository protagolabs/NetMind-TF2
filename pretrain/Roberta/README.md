# Intro
In this repo we pretrain [Roberta](https://arxiv.org/abs/1907.11692) by using the huggingface transofmers with Tensorflow 2.

We use [wikipedia](https://huggingface.co/datasets/wikipedia) and [bookcorpus](https://huggingface.co/datasets/bookcorpus) for training this model and evaluate the results on [GLUE](https://gluebenchmark.com/).

## Preprocessing ##

python **predata.py**

## Training ##

python **train.py**

## Evaluation ##

please run the bash files in the directory **run_glue_*.sh** 

## Result  ##

| GLUE       | CoLA | SST-2 | MRPC     | STS-B      | QQP       | MNLI      | QNLI | RTE  | WNLI  |
|------------|------|-------|------    | ----       |---        |---        | ---  | ---  | ---   |
| official   | 63.6 | 94.8  | 90.2     |91.2        |91.9       |87.6       | 92.8 | 78.7 | - |
| ours       | 62.34| 91.39 |87.92     |87.65       |87.35      |78.5       | 86.54| 64.25| - |



## Acknowledgement ##
[Huggingface Transformers Language Modeling](https://github.com/huggingface/transformers/blob/master/examples/tensorflow/language-modeling/run_mlm.py)

Thanks for my colleages Xiangpeng Wan and Yu Cheng for their kindly helps
