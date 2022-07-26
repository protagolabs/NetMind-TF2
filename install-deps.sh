# create netmind-tf2 env
conda create --name netmind-tf2 python=3.10

# load enviroment
conda activate netmind-tf2

# install the cuda and cudnn
conda install -c anaconda cudnn

# add cuda/cudnn to path, otherwise you will meet issue when running tensorflow as:
# Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory
nano ~/.bashrc

# add the following cuda/cudnn path to the end

export PATH=/home/xing/miniconda3/envs/netmind-tf2/lib:${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/home/xing/miniconda3/envs/netmind-tf2/lib:${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

# load again
source ~/.bashrc

conda activate netmind-tf2

# install tensorflow-gpu
pip install tensorflow-gpu

# enjoy!
