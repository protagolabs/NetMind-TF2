# Finetune_GPT2_Chatbot in Docker
In this repo we create a docker for finetuning the [GPT2 model](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) in tensorflow-2.x

This project is developed in docker-20.10.7

## Build ##

git clone 

## finetune ##
download [train](https://drive.google.com/file/d/1urLZaI8NlnQwQsH_dKPItDWcSyFqw4oP/view?usp=sharing) and [valid](https://drive.google.com/file/d/1g107ztO3fyf2Y-wEaZ6JkgdgM4WGvNxy/view?usp=sharing) dataset. Put them in current GPT2 directory.

**preparing data** python predata-tf2.py

**train**: python train-tf2.py

**multiworker-mirrored strategy distributed train**: python train-tf2-mm.py

**generate(test)**: python generate-tf2.py

## Acknowledgement ##
[Huggingface Transformers Language Modeling](https://github.com/huggingface/transformers/blob/master/examples/tensorflow/language-modeling/run_clm.py)

[Tensorflow Finetuning Bert](https://www.tensorflow.org/official_models/fine_tuning_bert) 

Thanks for my colleages Xiangpeng Wan and Yu Cheng for their kindly helps
