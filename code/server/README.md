# TensorRT Inference Server
> TLDR: A system which runs the TensorRT Inference Server by Nvidia with a few caveats.

## Overview
This constitutes the infrence portion of the Cloud Inference Benchmarking Suite.
The inference server is based on Nvidia’s TensorRT Inference Server and includes two additional components cAdvisor and Node exporter that aggregate and export the performance characteristics and resource utilization information such as GPU, CPU and network utilization of running containers.

## Configuration
The TensorRT server can be configured with any of the settings outlined in the original repository.
The main configuration for this component comes in the form of updating the `model_respository` or setting uncommenting the `runtime: nvidia` flag to enable GPU support.
 
## Usage
Starting
> docker-compose up -d 

Stopping
> docker-compose stop

Resetting
> docker-compose down --volumes --rmi 'all'

