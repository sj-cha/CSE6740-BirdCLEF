docker run -it -p 8888:8888 \
  -v $(pwd)/kaggle/input/birdclef-2024:/data/kaggle \
  -v $(pwd)/tf:/tf \
  --name cse6740-birdclef-container \
  cse6740-birdclef-jupyter-lab