# Bert
In this repo we pretrain [Bert](https://arxiv.org/abs/1810.04805) by using the huggingface transofmers with Tensorflow 2.

We use [wikipedia](https://huggingface.co/datasets/wikipedia) and [bookcorpus](https://huggingface.co/datasets/bookcorpus) for training this model and evaluate the results on [GLUE](https://gluebenchmark.com/).

## training ##

python **train.py**

## evaluation ##

please run the bash files in the directory **glue_eval** 

## Result  ##

| GLUE       | CoLA | SST-2 | MRPC     | STS-B      | QQP       | MNLI      | QNLI | RTE | WNLI  |
|------------|------|-------|------    | ----       |---        |---        | ---  | --- | ---   |
| HF official| 56.53| 92.32 |89.47/85.29|88.64/88.48|90.71/87.49|83.91      | 90.66| 65.7| 56.34 |
| ours       | 50.15| 90.60 |87.77/83.82|88.21/87.93|90.02/86.65|80.78/79.61| 88.88|63.17| 56.33 |



## Acknowledgement ##
[Huggingface Transformers Language Modeling](https://github.com/huggingface/transformers/blob/master/examples/tensorflow/language-modeling/run_mlm.py)

Thanks for my colleages Xiangpeng Wan and Yu Cheng for their kindly helps
