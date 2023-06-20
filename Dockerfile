# Start your image with a ubuntu base image
FROM ubuntu:22.04

# The /app directory should act as the main application directory
WORKDIR /app

# Copy the app package and package-lock.json file
# COPY package*.json ./

# # Copy local directories to the current local directory of our docker image (/app)
# COPY ./src ./src
# COPY ./public ./public
COPY ./code ./code
COPY ./data ./data
RUN chmod +rx ./code/*.py\
    && chmod +rx ./code/*/*.py

ENV PATH "./code/dba:$PATH"

# Install node packages, install serve, build the app, and remove dependencies at the end
RUN apt-get update\
    && apt-get -y install sudo\
    && apt-get upgrade -y

# dependencies
RUN sudo apt install python3.10\
    && sudo apt install python3-pip\
    && sudo apt install python3-venv\
    && sudo apt install git\
    && sudo apt install wget\
    && sudo apt install nvidia-525\
    && sudo apt-get install zlib1g

# CUDA
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb\
    && sudo dpkg -i cuda-keyring_1.0-1_all.deb\
    && sudo apt-get update\
    && sudo apt-get -y install cuda

# CUDNN
RUN sudo apt-get install libcudnn8=8.9.2.*-1+cuda12.1\
    && sudo apt-get install libcudnn8-dev=8.9.2.*-1+cuda12.1\
    && sudo apt-get install libcudnn8-samples=8.9.2.*-1+cuda12.1

# Python environment
RUN python3.10 -m venv jax-env\
    && source jax-env/bin/activate\
    && pip install --upgrade --no-cache-dir pip wheel\
    && pip install --upgrade --no-cache-dir numpy pandas scipy matplotlib wandb plotly\
    && pip install --upgrade --no-cache-dir "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html\
    && pip install --upgrade --no-cache-dir flax\
    && pip install --upgrade --no-cache-dir pyvista[all]

EXPOSE 3000

# Start the app
CMD [ "main.py","-h"]