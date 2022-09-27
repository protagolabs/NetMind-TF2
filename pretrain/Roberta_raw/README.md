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

## Improvements ##

In this project, we only use 4 datasets (bookcorpus, wikipedia, cc_news and openwebtext) due to the data access availability. In original paper, more data is used for training the model. 

parameter finetuning is needed for this project.

we only train this model on 5 epochs and it takes 3 weeks to finish the training on 4 x 3090 gpus.

## Acknowledgement ##
[Huggingface Transformers Language Modeling](https://github.com/huggingface/transformers/blob/master/examples/tensorflow/language-modeling/run_mlm.py)

Thanks for my colleages Xiangpeng Wan and Yu Cheng for their kindly helps
