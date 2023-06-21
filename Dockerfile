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

RUN pip install --upgrade pip wheel\
    && pip install --upgrade --no-cache-dir numpy pandas scipy matplotlib wandb plotly\
    && pip install --upgrade --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118\
    && pip install --upgrade --no-cache-dir "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html\
    && pip install --upgrade --no-cache-dir flax\
    && pip install --upgrade pyvista[all]

EXPOSE 3000

# Start the app
CMD ["nvidia-smi"]