NetMind-Models-TF2
We develop the netmind platform in tensorflow2

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
conda create --name tf2 python=3.9
```
* load enviroment
```bash
conda activate tf2
```

* install the cuda and cudnn
```bash
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
```

* add cuda/cudnn to path, otherwise you will meet issue when running tensorflow as: Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory
```bash
nano ~/.bashrc
```
* add the following cuda/cudnn path to the end, change "your_home_dir".
```
export PATH=/your_home_dir/miniconda3/envs/tf2/lib:${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/your_home_dir/miniconda3/envs/tf2/lib:${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

* load again
```bash
source ~/.bashrc
```
```bash
conda activate tf2
```
* install tensorflow-gpu
```bash
pip install tensorflow-gpu==2.9.1
```
#

# test your install
```bash
wget https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip
unzip -q kagglecatsanddogs_5340.zip
```

```python
python test_example.py
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
