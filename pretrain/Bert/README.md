# Bert
In this repo we pretrain [Bert](https://arxiv.org/abs/1810.04805) by using the huggingface transofmers with Tensorflow 2.

We use [wikipedia](https://huggingface.co/datasets/wikipedia) and [bookcorpus](https://huggingface.co/datasets/bookcorpus) for training this model and evaluate the results on [GLUE](https://gluebenchmark.com/).

## training ##

python **train.py**

## evaluation ##

please run the bash files in the directory **glue_eval** 

## Result 

GLUE | CoLA | #2 | #3 | #4 | #5 | #6 | #7 | #8 | #9 | #10 | #11
HF official | 56.53 | --- | --- |--- |--- |--- |--- |--- |--- |--- |---
ours | 50.15 | 283 | 290 | 286 | 289 | 285 | 287 | 287 | 272 | 276 | 269

CoLA (Matthews corr)|SST-2 (Accuracy)|MRPC (F1/Accuracy)	STS-B (Pearson/Spearman corr.)	QQP (Accuracy/F1)	MNLI (Mached acc./Mismatched acc.)	QNLI (Accuracy)	RTE (Accuracy)	WNLI (Accuracy)

56.53	92.32	89.47/85.29	88.64/88.48	90.71/87.49 83.91	      90.66	65.7	56.34

50.15	90.60	87.77/83.82	88.21/87.93	90.02/86.65	80.79/79.61	88.88	63.17	56.33


## Acknowledgement ##
[Huggingface Transformers Language Modeling](https://github.com/huggingface/transformers/blob/master/examples/tensorflow/language-modeling/run_mlm.py)

Thanks for my colleages Xiangpeng Wan and Yu Cheng for their kindly helps
