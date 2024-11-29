# Base image
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install \
    torch torchaudio \
    transformers datasets \
    librosa \
    scikit-learn \
    matplotlib \
    tqdm \
    jupyterlab

# Expose JupyterLab port
EXPOSE 8888

# Run JupyterLab
CMD ["jupyter-lab", "--ip=0.0.0.0", "--allow-root", "--no-browser"]
