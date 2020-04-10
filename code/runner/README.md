# TensorRT Inference Server Benchmark Suite
> TLDR: A program which designed to estimate the characteristics of the inference server

## Overview
The benchmarking client pre-processes the input data from a given directory, stores the pre-processed data in the client’s memory, and then generates and sends inference requests to the inference server.
The number of clients in conjunction with each client’s estimation capacity determines the peak throughput (λ) in inference requests per second for a given model.
The client is able to achieve an accurate estimation without using server-side statistics.

## Usage
```text
usage: benchmark.py [-h] [-v] [-a] [--streaming] [--infinite] [-m MODEL_NAME]
                    [--model-version MODEL_VERSION] [-b BATCH_SIZE]
                    [-c CLASSES] [--attempts ATTEMPTS]
                    [-s {NONE,INCEPTION,VGG}] [-u URL] [-p PORT]
                    [--percentile PERCENTILE] [--sla SLA] [-f FILEPATH]
                    [-t THREADS] [-q QPS] [--qps-delta QPS_DELTA]
                    [-i PROTOCOL]
optional arguments:
  -h, --help            show this help message and exit
  -v, --verbose         Enable verbose output
  -a, --asynchronous    Use asynchronous inference API
  --streaming           Use streaming inference API. The flag is only
                        available with gRPC protocol.
  --infinite            Run at desired throughput indefinitely.
  -m MODEL_NAME, --model-name MODEL_NAME
                        Name of model. Default is densenet_onnx
  --model-version MODEL_VERSION
                        The version of the model to query. Default is -1.
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        Batch size. Default is 1.
  -c CLASSES, --classes CLASSES
                        Number of class results to report. Default is 1.
  --attempts ATTEMPTS   Number of attempts to estimate the max throughput.
                        Default is 1.
  -s {NONE,INCEPTION,VGG}, --scaling {NONE,INCEPTION,VGG}
                        Type of scaling to apply to image pixels. Default is
                        NONE.
  -u URL, --url URL     Inference server URL. Default is localhost.
  -p PORT, --port PORT  Inference server Port. Default is 8001.
  --percentile PERCENTILE
                        The SLA percentile. Default is 95.
  --sla SLA             The SLA. Default is 1 second.
  -f FILEPATH, --filepath FILEPATH
                        The filepath of the image files to send to the server.
                        Default is /app/Dataset/.
  -t THREADS, --threads THREADS
                        Number of threads. Default is number of cpus times 16.
  -q QPS, --qps QPS     Number of queries to send per second. Default is 5.0.
  --qps-delta QPS_DELTA
                        The number of queries to send per second to increment
                        the test by. Default is 0.5.
  -i PROTOCOL, --protocol PROTOCOL
                        Protocol (HTTP/gRPC) used to communicate with
                        inference service. Default is gRPC.
```
