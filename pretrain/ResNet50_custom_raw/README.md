This repo is for pretraining ResNet50 on [Imagenet-1K](https://image-net.org/) 

## 1. data preparing ##

1. please download the train and val datasets from [Imagenet-1K](https://image-net.org/) （you may need to create an account to access the dataset, or check the local disk）

2. after download the datasets (ILSVRC2012_img_train.tar and ILSVRC2012_img_val.tar), please run the script as 

```
bash extract_ILSVRC.sh
```
3. change the data directory in config.py


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


