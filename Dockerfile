FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04

ENV PYTHON_VERSION=3.10.5

RUN export DEBIAN_FRONTEND=noninteractive \
    && apt-get update \
    && apt-get -y install curl \
    && apt -y update --no-install-recommends \
    && apt -y install --no-install-recommends \
    gcc-9 g++-9 wget git build-essential \
    libatlas-base-dev \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-regex-dev \
    libboost-system-dev \
    libboost-test-dev \
    libeigen3-dev \
    libsuitesparse-dev \
    libmetis-dev \
    libfreeimage-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libcgal-qt5-dev \
    libffi-dev libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
    libncurses5-dev libncursesw5-dev xz-utils tk-dev \
    xorg-dev libglu1-mesa-dev -y \
    && apt autoremove -y \
    && apt clean -y

# setup GCC
RUN \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 20 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 20 \
    && update-alternatives --install /usr/bin/cc cc /usr/bin/gcc 30 \
    && update-alternatives --set cc /usr/bin/gcc \
    && update-alternatives --install /usr/bin/c++ c++ /usr/bin/g++ 30 \
    && update-alternatives --set c++ /usr/bin/g++

# install python
RUN \
    wget https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz \
    && tar xzf Python-${PYTHON_VERSION}.tgz \
    && cd ./Python-${PYTHON_VERSION} \
    && ./configure --enable-optimizations \
    && make -j \
    && make install \
    && python3 -m pip install --upgrade pip setuptools wheel cmake \
    && echo "alias pip=pip3" >> ~/.bashrc \
    && sh ~/.bashrc \


ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN apt-get update

RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh \
    && bash miniconda.sh -b -p /opt/conda \
    && rm -f miniconda.sh

# Initialize Conda and set up environment
RUN echo "export PATH=/opt/conda/bin:$PATH" >> ~/.bashrc

ENV PATH="/opt/conda/bin:${PATH}"

RUN conda init bash

RUN conda create -n NeuralPointLF python=3.7 \
    && eval "$(conda shell.bash hook)" \
    && conda activate NeuralPointLF \
    && echo "conda activate NeuralPointLF" >> ~/.bashrc \
    && conda activate NeuralPointLF \
    && conda install -c pytorch -c conda-forge pytorch=1.7.1 torchvision=0.8.2 cudatoolkit=11.0 \
    && conda install -c fvcore -c iopath -c conda-forge fvcore iopath \
    && conda install -c bottler nvidiacub \
    && conda install jupyterlab \
    && pip install scikit-image matplotlib imageio plotly opencv-python \
    && conda install -c pytorch3d pytorch3d \
    && conda install -c open3d-admin -c conda-forge open3d \
    && conda clean -afy

# Set working directory
WORKDIR /content

RUN apt-get autoremove -y && apt-get clean -y

RUN pip install --upgrade git+https://github.com/haven-ai/haven-ai

# boost path
RUN mkdir /include && ln -s /usr/include/boost /include/boost

# build content path
RUN mkdir -p /content
WORKDIR /content


RUN apt-get update && apt-get install -y git

CMD ["/bin/bash"]
