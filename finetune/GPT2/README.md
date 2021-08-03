# GPT2
In this repo we develop a chatbot by finetuning the GPT2 model in tensorflow 2.x

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
download [train](https://drive.google.com/file/d/1urLZaI8NlnQwQsH_dKPItDWcSyFqw4oP/view?usp=sharing) and [valid](https://drive.google.com/file/d/1g107ztO3fyf2Y-wEaZ6JkgdgM4WGvNxy/view?usp=sharing) dataset

**preparing data** python predata-tf2.py

**train**: python train-tf2.py

**multiworker-mirroredstrategy distributed train**: python train-tf2-mm.py

**generate(test)**: python generate-tf2.py

## Acknowledgement ##
[Huggingface Transformers Language Modeling](https://github.com/huggingface/transformers/blob/master/examples/tensorflow/language-modeling/run_clm.py)

[Tensorflow Finetuning Bert](https://www.tensorflow.org/official_models/fine_tuning_bert) 

Thanks for my colleages Xiangpeng Wan and Yu Cheng for their kindly helps
