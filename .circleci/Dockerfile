FROM python:3.12.0-slim

# Avoid Python writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Remove any third-party apt sources to avoid issues with expiring keys.
RUN rm -f /etc/apt/sources.list.d/*.list

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
 && rm -rf /var/lib/apt/lists/*

# Create a clean virtual environment
# and add venv to PATH so it's always active
ENV VIRTUAL_ENV=/venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Upgrade pip/setuptools/wheel inside the venv
RUN pip install --upgrade pip setuptools wheel

## Install mlcg v0.1.1
WORKDIR /tmp/build
RUN git clone https://github.com/ClementiGroup/mlcg.git mlcg
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r mlcg/env_cpu_with_hashes.in
RUN pip install --no-deps git+https://github.com/ACEsuit/mace.git@v0.3.13
RUN pip install --no-deps nequip==0.12.1 nequip-allegro==0.7.0
RUN pip install mlcg/.
RUN rm -rf /tmp/build

# Create a working directory
RUN mkdir /app
WORKDIR /app

# Default command (gets you a shell with venv active)
CMD ["bash"]