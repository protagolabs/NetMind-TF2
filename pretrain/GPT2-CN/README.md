This repo is for pretraining GPT2 on [CLUECorpus2020](https://github.com/CLUEbenchmark/CLUECorpus2020/) 

## 1. data preparing ##

1. please download all the processed datas from [here](https://github.com/CLUEbenchmark/CLUECorpus2020/) and unzip.

2. run the code as below, make sure the data dir is correct in preCLUEdata.py

```python
python preCLUEdata.py
```

3. You will see the generated data "lm_dataset_length128_chuck512" after a while.


## Training and Evaluating ##

* For training on one single gpu on one node, please run the script：
```
python train.py
```
The pretrained weights and training results could be found in ./result 

* For training on multi-gpus on one node with eager mode , please run the script：
```
python train_mm.py
```

* For training on multi-gpus on one node with trainer , please run the script：
```
python train_trainer.py
```

** when running the trainer, the training sometimes is stucked at the end of epoch. will fix it later 


## Result ##

we meet the result in [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

The result on ImageNet-1k is:

|ImageNet-1k|top-1 | top-5|
|---        |---   |---   |
|      single GPU     |75.80 |92.95 |
|      Multi-GPU(4 gpus)     |TBD |TBD|


