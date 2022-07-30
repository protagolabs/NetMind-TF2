This repo is for pretraining ResNet50 on [Food101 dataset](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) 

## 1. data preparing 

1. please download the food101 dataset from the [website](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) or through 

```bash
wget http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz
```

2. unzip the dataset
```bash
tar xzvf food-101.tar.gz
```
3. delete the downloaded file
```bash
rm food-101.tar.gz
```

## 

## Distributed Training and Evaluating 

```bash
bash local_run_trainer.sh
```

##


