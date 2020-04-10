# TensorRT Inference Monitor
> TLDR: A system which uses Prometheus to monitor the inference server with a Grafana dashboard

## Overview
This constitutes the monitoring portion of the Cloud Inference Benchmarking Suite.
This component of the suite uses Prometheus to scrape the cadvisor, node-exporter, and tensorRT Inference Server metrics from the server hosting the models.
Then, it uses a Grafana dashboard to display the metrics generated by the server.
The visualizations can be viewed at `http://<SERVER IP>:3000/`.
Withe current configuration the username is `admin` and the password is `pass`, but this can be modified using the `${ADMIN_USER:-admin}` and `${ADMIN_PASSWORD:-pass}` fields.

## Configuration
In to gather the TensorRT inference Server's Metrics, Prometheus needs to be pointed to the server's IP address. Edit the `target` field of each of the jobs with the `tensorrt-` prefix in the `prometheus/prometheus.yml` file to point to the server's IPv4 address.
 
## Usage
Starting
> docker-compose up -d 

Stopping
> docker-compose stop

Resetting
> docker-compose down --volumes