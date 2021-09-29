#FROM nvidia/cuda:11.1.1-runtime-ubuntu20.04
FROM nvidia/cuda:11.1.1-runtime-ubuntu20.04
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
ARG LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"


RUN apt-get update \
    && apt-get install -y python3-dev wget apt-utils 

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir root/.conda \
    && sh Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh

RUN conda create -y -n ml python=3.8

COPY . src/
# RUN /bin/bash -c "cd src \
#     && source activate ml \
#     && conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia \
#     && pip install -r requirements.txt"

RUN /bin/bash -c "cd src \
   && source activate ml \
   && cp cuda/lib64/* /usr/local/cuda/lib64 \
   && cp cuda/include/* /usr/local/cuda/include \
   && pip install -r requirements.txt"