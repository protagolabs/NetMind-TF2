This repo is for pretraining ResNet50 on [Imagenet-1K](https://image-net.org/) （you may need to create an account to access the dataset, or check the local disk）


## data preparing ##

please download the 

we meet the result in [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

The result on ImageNet-1k is:

|ImageNet-1k|top-1 | top-5|
|---        |---   |---   |
|      single GPU     |75.80 |92.95 |
|      Multi-GPU(2 gpus)     |TBD |TBD|


The pretrained weights and training results could be found in ./result 
