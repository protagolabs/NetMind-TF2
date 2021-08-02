# NetMind-Models-TF2
In this repo we develop a chatbot by pretraining/finetuning the GPT2 model in tensorflow 2.x

This project is developed in tensorflow-gpu=2.5.0, cuda=11.2, and transformers==4.8.1

## installation ##
conda create --name xxx python=3.8

pip install tensorflow-gpu==2.5.0

conda install pandas

conda install matplotlib

pip install transformers

pip install tf-models-official


## finetune ##
download [train](https://drive.google.com/file/d/1urLZaI8NlnQwQsH_dKPItDWcSyFqw4oP/view?usp=sharing) and [valid](https://drive.google.com/file/d/1g107ztO3fyf2Y-wEaZ6JkgdgM4WGvNxy/view?usp=sharing) dataset into the directory ./finetune

cd ./finetune

python predata.py

**train**: python train.py

**generate(test)**: python generate.py

## Acknowledgement ##
[Huggingface Transformers](https://github.com/huggingface/transformers/blob/master/examples/tensorflow/language-modeling/run_clm.py)
Thanks for my colleages Xiangpeng Wan and Yu Cheng for their kindly helps
