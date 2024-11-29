# CSE6740-BirdCLEF
Final project for CSE 6740 at Georgia Tech

# Quick Start
##  Pull Docker Image
`docker-compose up` will pull a 3.67 GB tensorflow docker image (v2.16.2) with CUDA kit and cudNN packaged together.
## Add More Packages
edit this line `RUN pip install -U xxx` in Dockerfile