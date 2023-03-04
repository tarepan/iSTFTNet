# derived from Microsoft Python dev container (https://github.com/microsoft/vscode-dev-containers/tree/main/containers/python-3)
FROM mcr.microsoft.com/vscode/devcontainers/python:3.8

# Colab versions @2023-03-04
# Python:                 3.8.10
# NumPy:      numpy       1.22.4
# PyTorch:    torch       1.13.1+cu116
# Torchaudio: torchaudio  0.13.1+cu116

# Latest Numpy (old numpy has typing issues)
RUN pip3 --disable-pip-version-check --no-cache-dir install numpy

# PyTorch on CPU
RUN pip3 --disable-pip-version-check --no-cache-dir install torch==1.13.1+cpu torchaudio==0.13.1+cpu -f https://download.pytorch.org/whl/torch_stable.html

# package dependency manager `poetry` without environment isolation (It is a job of Docker.)
ENV POETRY_VIRTUALENVS_CREATE=false
RUN curl -sSL https://install.python-poetry.org | python3 -