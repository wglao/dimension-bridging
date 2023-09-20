# Start your image with a ubuntu base image
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# The /app directory should act as the main application directory
WORKDIR /app

# Install node packages, install serve, build the app, and remove dependencies at the end
RUN apt update\
    && apt -y install sudo\
    && apt upgrade -y

# dependencies
RUN sudo apt -y update\
    && sudo apt install -y software-properties-common curl wget tar build-essential git htop libxrender1

# python
RUN sudo apt install -y python3.10 python3-dev python3-doc python3-pip python3-venv\
    && ln -s /usr/bin/python3.10 /usr/bin/python

RUN sudo apt update\
    && sudo apt upgrade -y\
    && sudo apt clean\
    && rm -rf /var/lib/apt/lists/*\
    && sudo apt autoremove

# venv
RUN python3.10 -m venv jax-env

ENV VIRTUAL_ENV "/app/jax-env/"
ENV PATH "/app/jax-env/bin:$PATH"

# python packages
RUN pip install --upgrade pip wheel\
    && pip install --upgrade --no-cache-dir numpy pandas scipy matplotlib wandb plotly\
    && pip install --upgrade --no-cache-dir networkx[default] metis\
    && pip install --upgrade --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118\
    && pip install --upgrade --no-cache-dir torch_geometric\
    && pip install --upgrade --no-cache-dir pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html\
    && pip install --upgrade --no-cache-dir "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html\
    && pip install --upgrade --no-cache-dir flax\
    && pip install --upgrade pyvista[all]

# metis
RUN cd /app && wget http://glaros.dtc.umn.edu/gkhome/fetch/sw/metis/metis-5.1.0.tar.gz\
    && gunzip metis-5.1.0.tar.gz\
    && tar -xvf metis-5.1.0.tar\
    && rm metis-5.1.0.tar\
    && cd metis-5.1.0\
    && sed -i 's/IDXTYPEWIDTH 32/IDXTYPEWIDTH 64/g' include/metis.h\
    && sed -i 's/REALTYPEWIDTH 32/REALTYPEWIDTH 64/g' include/metis.h\
    && make config prefix=/app/metis-5.1.0 shared=1\
    && make install

ENV METIS_DLL "/app/metis-5.1.0/lib/libmetis.so"
ENV METIS_IDXTYPEWIDTH "64"
ENV METIS_REALTYPEWIDTH "64"


EXPOSE 3000

# Start the app
CMD ["nvidia-smi"]