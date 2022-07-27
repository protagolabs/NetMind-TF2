NetMind-Models-TF2
We develop the netmind platform in tensorflow 2

# install conda

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

```bash
bash Miniconda3-latest-Linux-x86_64.sh
```

```bash
source ~/.bashrc
```

```bash
conda update --force conda
```
#

# install the tensorflow and some other deps
* create netmind-tf2 env
```bash
conda create --name netmind-tf2 python=3.10
```
* load enviroment
```bash
conda activate netmind-tf2
```

* install the cuda and cudnn
```bash
conda install -c anaconda cudnn
```

* add cuda/cudnn to path, otherwise you will meet issue when running tensorflow as: Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory
```bash
nano ~/.bashrc
```
* add the following cuda/cudnn path to the end, change "your_home_dir".
```
export PATH=/your_home_dir/xing/miniconda3/envs/netmind-tf2/lib:${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/your_home_dir/miniconda3/envs/netmind-tf2/lib:${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

* load again
```bash
source ~/.bashrc
```
```bash
conda activate netmind-tf2
```
* install tensorflow-gpu
```bash
pip install tensorflow-gpu
```
#

# Enjoy different categories could be found in corresponding directory: 

pretrain 

finetune

etc..
#

## Acknowledgement ##
[Huggingface Transformers Language Modeling](https://github.com/huggingface/transformers/blob/master/examples/tensorflow/language-modeling/run_clm.py)

[Tensorflow](https://www.tensorflow.org/) 

Thanks for my colleages Xiangpeng Wan and Yu Cheng for their kindly helps
