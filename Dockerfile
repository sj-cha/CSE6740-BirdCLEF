# This assumes the container is running on a system with a CUDA GPU
FROM tensorflow/tensorflow:2.16.2-gpu-jupyter

WORKDIR /tf

RUN pip install -U jupyterlab pandas matplotlib

EXPOSE 8888

ENTRYPOINT ["jupyter", "lab","--ip=0.0.0.0","--allow-root","--no-browser"]
