# NetMind-Models-TF2
In this repo we develop the netmind platform in tensorflow 2.x

This project is developed in tensorflow-gpu=2.5.0, cuda=11.2, and transformers==4.8.1

## installation ##
conda create --name xxx (the env name you wish) python=3.8

conda activate xxx

pip install tensorflow-gpu==2.5.0

conda install pandas

conda install matplotlib

pip install transformers

pip install tf-models-official


## finetune ##
check the models in ./finetune

## Acknowledgement ##
[Huggingface Transformers Language Modeling](https://github.com/huggingface/transformers/blob/master/examples/tensorflow/language-modeling/run_clm.py)

[Tensorflow Finetuning Bert](https://www.tensorflow.org/official_models/fine_tuning_bert) 

Thanks for my colleages Xiangpeng Wan and Yu Cheng for their kindly helps
