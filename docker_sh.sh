#!/bin/bash
docker build -t  cuda_16_10_devel --build-arg CACHEBUST=$(date +%s) . 
docker rm test_cuda_1
docker run --runtime=nvidia -ti --name=test_cuda_1 cuda_16_10_devel
