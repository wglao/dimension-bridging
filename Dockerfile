# Start your image with a ubuntu base image
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# The /app directory should act as the main application directory
WORKDIR /app

# Install node packages, install serve, build the app, and remove dependencies at the end
RUN apt update\
    && apt -y install sudo\
    && apt upgrade -y

# python
RUN sudo apt -y install python3.10\
    && sudo apt -y install python3-pip\
    && sudo apt -y install python3-venv

# Python environment
RUN python3.10 -m venv jax-env

ENV VIRTUAL_ENV "/app/jax-env/"
ENV PATH "/app/jax-env/bin:$PATH"

RUN pip install --upgrade pip wheel\
    && pip install --upgrade numpy pandas scipy matplotlib wandb plotly\
    && pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html\
    && pip install --upgrade flax\
    && pip install --upgrade pyvista[all]

EXPOSE 3000

# Start the app
CMD ["nvidia-smi","&&","/app/jax-env/bin/python -c 'import jax; jax.devices()'"]