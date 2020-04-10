#!/usr/bin/env python
import csv
import datetime
from argparse import ArgumentParser
from builtins import range
from functools import partial
from multiprocessing import Process, cpu_count, Pool
from os import listdir
from time import time, sleep

import numpy as np
import tensorrtserver.api.model_config_pb2 as model_config
from PIL import Image
from tensorrtserver.api import InferContext
from tensorrtserver.api import ProtocolType
from tensorrtserver.api import ServerStatusContext


class ImageBasedModel(Process):
    """
    A class for pre-processing Images for inference
    """

    @staticmethod
    def model_dtype_to_np(model_dtype):
        """
        Converts a model data type to a numpy data type
        # Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
        #
        # Redistribution and use in source and binary forms, with or without
        # modification, are permitted provided that the following conditions
        # are met:
        #  * Redistributions of source code must retain the above copyright
        #    notice, this list of conditions and the following disclaimer.
        #  * Redistributions in binary form must reproduce the above copyright
        #    notice, this list of conditions and the following disclaimer in the
        #    documentation and/or other materials provided with the distribution.
        #  * Neither the name of NVIDIA CORPORATION nor the names of its
        #    contributors may be used to endorse or promote products derived
        #    from this software without specific prior written permission.
        #
        # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
        # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
        # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
        # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
        # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
        # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
        # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
        # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
        # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
        # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
        # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
        @param model_dtype: Model data type
        @return: Numpy data type
        """
        if model_dtype == model_config.TYPE_BOOL:
            return np.bool
        elif model_dtype == model_config.TYPE_INT8:
            return np.int8
        elif model_dtype == model_config.TYPE_INT16:
            return np.int16
        elif model_dtype == model_config.TYPE_INT32:
            return np.int32
        elif model_dtype == model_config.TYPE_INT64:
            return np.int64
        elif model_dtype == model_config.TYPE_UINT8:
            return np.uint8
        elif model_dtype == model_config.TYPE_UINT16:
            return np.uint16
        elif model_dtype == model_config.TYPE_FP16:
            return np.float16
        elif model_dtype == model_config.TYPE_FP32:
            return np.float32
        elif model_dtype == model_config.TYPE_FP64:
            return np.float64
        elif model_dtype == model_config.TYPE_STRING:
            return np.dtype(object)
        return None

    @staticmethod
    def parse_model(url: str, protocol: ProtocolType, model_name: str, batch_size: int, verbose=False):
        """
        Determines a model's configuration from by interpreting the results of Nvidia's TenorRT Inference Server

        # Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
        #
        # Redistribution and use in source and binary forms, with or without
        # modification, are permitted provided that the following conditions
        # are met:
        #  * Redistributions of source code must retain the above copyright
        #    notice, this list of conditions and the following disclaimer.
        #  * Redistributions in binary form must reproduce the above copyright
        #    notice, this list of conditions and the following disclaimer in the
        #    documentation and/or other materials provided with the distribution.
        #  * Neither the name of NVIDIA CORPORATION nor the names of its
        #    contributors may be used to endorse or promote products derived
        #    from this software without specific prior written permission.
        #
        # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
        # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
        # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
        # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
        # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
        # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
        # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
        # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
        # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
        # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
        # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

        @param url: The server's url
        @param protocol: The protocol used to access the server (i.e. gRPC or REST)
        @param model_name: The name of the model
        @param batch_size: The desired model batch size
        @param verbose: If true, prints out the server's response
        @return:
        """
        ctx = ServerStatusContext(url, protocol, model_name, verbose)
        server_status = ctx.get_server_status()

        if model_name not in server_status.model_status:
            raise Exception("unable to get status for '" + model_name + "'")

        status = server_status.model_status[model_name]
        config = status.config

        if len(config.input) != 1:
            raise Exception("expecting 1 input, got {}".format(len(config.input)))
        if len(config.output) != 1:
            raise Exception("expecting 1 output, got {}".format(len(config.output)))

        input = config.input[0]
        output = config.output[0]

        if output.data_type != model_config.TYPE_FP32:
            raise Exception("expecting output datatype to be TYPE_FP32, model '" +
                            model_name + "' output type is " +
                            model_config.DataType.Name(output.data_type))

        # Output is expected to be a vector. But allow any number of dimensions as long as all but 1 is size 1
        # (e.g. { 10 }, { 1, 10}, { 10, 1, 1 } are all ok). Variable-size dimensions are not currently supported.
        non_one_cnt = 0
        for dim in output.dims:
            if dim == -1:
                raise Exception("variable-size dimension in model output not supported")
            if dim > 1:
                non_one_cnt += 1
                if non_one_cnt > 1:
                    raise Exception("expecting model output to be a vector")

        # Model specifying maximum batch size of 0 indicates that batching is not supported and so the input tensors do
        # not expect an "N" dimension (and 'batch_size' should be 1 so that only a single image instance is inferred at
        # a time).
        max_batch_size = config.max_batch_size
        if max_batch_size == 0:
            if batch_size != 1:
                raise Exception("batching not supported for model '" + model_name + "'")
        else:  # max_batch_size > 0
            if batch_size > max_batch_size:
                raise Exception("expecting batch size <= {} for model {}".format(max_batch_size, model_name))

        # Model input must have 3 dims, either CHW or HWC
        if len(input.dims) != 3:
            raise Exception(
                "expecting input to have 3 dimensions, model '{}' input has {}".format(
                    model_name, len(input.dims)))

        # Variable-size dimensions are not currently supported.
        for dim in input.dims:
            if dim == -1:
                raise Exception("variable-size dimension in model input not supported")

        if ((input.format != model_config.ModelInput.FORMAT_NCHW) and
                (input.format != model_config.ModelInput.FORMAT_NHWC)):
            raise Exception("unexpected input format " + model_config.ModelInput.Format.Name(input.format) +
                            ", expecting " +
                            model_config.ModelInput.Format.Name(model_config.ModelInput.FORMAT_NCHW) +
                            " or " +
                            model_config.ModelInput.Format.Name(model_config.ModelInput.FORMAT_NHWC))

        if input.format == model_config.ModelInput.FORMAT_NHWC:
            h = input.dims[0]
            w = input.dims[1]
            c = input.dims[2]
        else:
            c = input.dims[0]
            h = input.dims[1]
            w = input.dims[2]

        return input.name, output.name, c, h, w, input.format, ImageBasedModel.model_dtype_to_np(input.data_type)

    @staticmethod
    def preprocess(img, input_format, data_type, c, h, w, scaling):
        """
        Pre-process an image to meet the size, type and format requirements specified by the parameters.

        # Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
        #
        # Redistribution and use in source and binary forms, with or without
        # modification, are permitted provided that the following conditions
        # are met:
        #  * Redistributions of source code must retain the above copyright
        #    notice, this list of conditions and the following disclaimer.
        #  * Redistributions in binary form must reproduce the above copyright
        #    notice, this list of conditions and the following disclaimer in the
        #    documentation and/or other materials provided with the distribution.
        #  * Neither the name of NVIDIA CORPORATION nor the names of its
        #    contributors may be used to endorse or promote products derived
        #    from this software without specific prior written permission.
        #
        # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
        # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
        # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
        # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
        # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
        # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
        # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
        # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
        # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
        # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
        # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

        @param img: The image to scale
        @param input_format: The image's format
        @param data_type: The image's desired data type
        @param c: The number of color channels
        @param h: The high of the image
        @param w: The width of the image
        @param scaling: The method used to scale the image (i.e. VGG, Inception, None)
        @return: The pre-processed image
        """
        if c == 1:
            sample_img = img.convert('L')
        else:
            sample_img = img.convert('RGB')

        resized_img = sample_img.resize((w, h), Image.BILINEAR)
        resized = np.array(resized_img)
        if resized.ndim == 2:
            resized = resized[:, :, np.newaxis]

        typed = resized.astype(data_type)

        if scaling == 'INCEPTION':
            scaled = (typed / 128) - 1
        elif scaling == 'VGG':
            if c == 1:
                scaled = typed - np.asarray((128,), dtype=data_type)
            else:
                scaled = typed - np.asarray((123, 117, 104), dtype=data_type)
        else:
            scaled = typed

        # Swap to CHW if necessary
        if input_format == model_config.ModelInput.FORMAT_NCHW:
            ordered = np.transpose(scaled, (2, 0, 1))
        else:
            ordered = scaled

        # Channels are in RGB order. Currently model configuration data doesn't provide any information as to other
        # channel orderings (like BGR) so we just assume RGB.
        return ordered

    def __del__(self):
        print("Thread Deleted")


class ModelBenchmarkRunner:
    """
    Ingests the data for inference and performs benchmarking
    """

    def __init__(self, flags):
        """
        Instantiate the benchmarking class
        @param flags: The flags which determine the benchmark client's execution
        """
        # The command line arguments
        self.flags = flags
        # An array containing all of the unprocessed images for inference
        self.raw_images = []
        # An array containing all of the unprocessed images for inference
        self.processed_images = []
        # The pre-batched set of images
        self.batches = []
        # An array of arrays containing each of the request latencies for a given throughput
        self.results = []
        # The request latencies for a given throughput
        self.output = []
        # A pool of processes to execute the inference requests with
        self.pool = Pool(processes=self.flags.threads)
        # Number of independent attempt to estimate the maximum latency
        self.attempts = 1
        # The starting initial starting throughput for performing inference
        self.original_qps = self.flags.qps

    def setup(self):
        """
        Loads and processes a data set
        """
        self.__check_settings()
        self.__load_raw_images()
        self.__pre_process_images()
        self.__batch_requests()

    def run_benchmark(self):
        # Try block which catches keyboard interrupts and allows for safely saving the results of a benchmark
        try:
            # While the peak throughput has not been estimated a given number of times
            while self.attempts <= self.flags.attempts:
                # Clear the output
                self.output = []
                # Reset the throughput
                self.flags.qps = self.original_qps

                # The number of times throughput has changed
                pass_number = 1
                # The state of estimating the maximum throughput
                max_throughput_met = False
                # The throughput estimated from the previous trail
                previous_throughput = -1.0
                # The number of passes  which have occurred since and not exceeded the previous throughput
                passes_since_max_throughput = 0

                # Check if detailed output should be printed
                if self.flags.verbose:
                    print(f'Attempt {self.attempts}: started')

                # While the peak throughput for a given attempt has not been reached
                while not max_throughput_met:

                    # Check if detailed output should be printed
                    if self.flags.verbose:
                        print(f'Attempt {self.attempts} Pass {pass_number}: started')

                    # The starting time of the estimation process
                    pass_start_time = time()

                    # For each of the batched images
                    for batch in self.batches:

                        # The start time of sending a request
                        start_time = time()

                        # If the request should be sent asynchronously
                        if self.flags.asynchronous:
                            self.pool.apply_async(ModelBenchmarkRunner.infer,
                                                  args=(self.flags, self.input_name, self.output_name, batch),
                                                  callback=partial(ModelBenchmarkRunner.inference_callback,
                                                                   self.results))
                        else:
                            self.pool.apply(ModelBenchmarkRunner.infer,
                                            args=(self.flags, self.input_name, self.output_name, batch))

                        # The amount of time to wait before sending another request, which is based on the qps
                        remaining_time = start_time + (1 / self.flags.qps) - time()
                        if remaining_time > 0.0:
                            sleep(remaining_time)

                    # Check if response has been received for all batches
                    while len(self.results) != len(self.batches):
                        pass

                    # Get the end time of a pass
                    pass_end_time = time()

                    # Estimate key characteristics of a pass
                    throughput = float(len(self.batches)) / (pass_end_time - pass_start_time)
                    custom_percentile = np.percentile(self.results, self.flags.percentile)

                    # Check if detailed output should be printed
                    if self.flags.verbose:
                        self.print_results(pass_number, throughput, pass_start_time, pass_end_time, custom_percentile)

                    # Add the results to the array of results
                    self.output.append(self.results)
                    self.results = []

                    # Increment the pass number
                    pass_number += 1

                    # If a benchmark is suppose to run at a constant rate indefinitely
                    if self.flags.infinite and self.flags.qps_delta == 0.0:
                        max_throughput_met = False
                    # If a benchmark is suppose to run at an increasing rate
                    elif not self.flags.infinite:
                        # Check if the peak throughput has been met or an SLA violation has occurred
                        if throughput > previous_throughput or custom_percentile > self.flags.sla:
                            previous_throughput = throughput
                            self.flags.qps += self.flags.qps_delta
                            passes_since_max_throughput = 0
                        else:
                            passes_since_max_throughput += 1
                            if passes_since_max_throughput > 5:
                                max_throughput_met = True
                    # If a benchmark is suppose to run at an increasing rate indefinitely
                    else:
                        if custom_percentile > self.flags.sla:
                            passes_since_max_throughput += 1
                            print(f'SLA violation # {passes_since_max_throughput}')
                            if passes_since_max_throughput == 5:
                                max_throughput_met = True
                        else:
                            passes_since_max_throughput = 0

                # Write the results to the file once the benchmarking phase has concluded for an attempt
                with open(self.flags.filepath + f'results_{self.attempts}.csv', "w+") as f:
                    writer = csv.writer(f, delimiter=',')
                    writer.writerows(self.output)

                print(f'Attempt {self.attempts}: Peak throughput {previous_throughput} at {datetime.datetime.now()}')

                # Increment the number of attempts
                self.attempts += 1

        # If the program is to be terminated early
        except KeyboardInterrupt:
            print('Program interrupted')
            with open(self.flags.filepath + f'results_{self.attempts}.csv', "w+") as f:
                writer = csv.writer(f, delimiter=',')
                writer.writerows(self.output)

        # Remove processes from pool
        self.pool.close()
        self.pool.join()

    @staticmethod
    def infer(flags, input_name, output_name, batch):
        """
        Perform inference non-asynchronosly
        @param flags: The server flags
        @param input_name: The name of the input field
        @param output_name: The name of the output field
        @param batch: The batch of images to send
        @return: The elapsed time for perfroming an inference request
        """
        inference_context = InferContext((flags.url + ":" + str(flags.port)), flags.protocol, flags.model_name,
                                         flags.model_version, False, 0, flags.streaming, [])

        start_time = time()

        inference_context.run({input_name: batch},
                              {output_name: (InferContext.ResultFormat.CLASS, flags.classes)},
                              flags.batch_size)

        end_time = time()

        return end_time - start_time

    def print_results(self, pass_number, throughput, pass_start_time, pass_end_time, custom_percentile):
        """
        Compute and print the results from a given pass
        """
        twenty_fifth_percentile = np.percentile(self.results, 25)
        fiftieth_percentile = np.percentile(self.results, 50)
        seventy_fifth_percentile = np.percentile(self.results, 75)

        print(f'Attempt {self.attempts} Pass {pass_number}: total batches {len(self.batches)}')
        print(f'Attempt {self.attempts} Pass {pass_number}: elapsed time {pass_end_time - pass_start_time} s')
        print(f'Attempt {self.attempts} Pass {pass_number}: desired throughput {self.flags.qps} qps')
        print(f'Attempt {self.attempts} Pass {pass_number}: actual throughput {throughput} qps')
        print(f'Attempt {self.attempts} Pass {pass_number}: 25th-percentile {twenty_fifth_percentile} s')
        print(f'Attempt {self.attempts} Pass {pass_number}: 50th-percentile {fiftieth_percentile} s')
        print(f'Attempt {self.attempts} Pass {pass_number}: 75th-percentile {seventy_fifth_percentile} s')
        print(f'Attempt {self.attempts} Pass {pass_number}: {self.flags.percentile}th-percentile {custom_percentile} s')
        print(f'Attempt {self.attempts} Pass {pass_number}: completed at {datetime.datetime.now()}')

    @staticmethod
    def inference_callback(results, elapsed_time):
        """
        A callback function which handles the results of a asynchronous inference request
        @param results: The place to put the asynchronous request
        @param elapsed_time: The amount of required for the inference request to complete
        """
        results.append(elapsed_time)

    def __check_settings(self):
        """
        Examine the settings to verify that they are not conflicting
        """
        # Check if protocol and flag combination is valid
        self.flags.protocol = ProtocolType.from_str(self.flags.protocol)
        if self.flags.streaming and self.flags.protocol != ProtocolType.GRPC:
            raise Exception("Streaming is only allowed with gRPC protocol")

    def __load_raw_images(self):
        """
        Load the images from their respective files into memory by copying the image
        """
        self.filenames = listdir(self.flags.filepath)
        for filename in self.filenames:
            with open(self.flags.filepath + filename, 'rb') as i:
                img = Image.open(i).copy()
                self.raw_images.append(img)

    def __pre_process_images(self):
        """
        Adjust the formatting of the loaded images
        """
        # Make sure the model matches our requirements, and get some properties of the model that we need for
        # pre-processing
        self.input_name, self.output_name, self.c, self.h, self.w, self.format, self.dtype = \
            ImageBasedModel.parse_model(
                (self.flags.url + ":" + str(self.flags.port)),
                self.flags.protocol,
                self.flags.model_name,
                self.flags.batch_size,
                False)

        # Pre-process the images into input data according to model requirements
        for raw_image in self.raw_images:
            self.processed_images.append(
                ImageBasedModel.preprocess(raw_image, self.format, self.dtype, self.c, self.h, self.w,
                                           self.flags.scaling))

        self.raw_images = []

    def __batch_requests(self):
        """
        Batch all of the processed images before performing inference
        """
        image_idx = 0
        last_request = False
        while not last_request:
            input_batch = []
            for idx in range(self.flags.batch_size):
                input_batch.append(self.processed_images[image_idx])
                image_idx = (image_idx + 1) % len(self.processed_images)
                if image_idx == 0:
                    last_request = True
            self.batches.append(input_batch)

        self.processed_images = []


# If this file is run as a standalone program
if __name__ == '__main__':
    # Collect the arguments that that define the program's behavior
    parser = ArgumentParser()
    parser.add_argument('-v', '--verbose', action="store_true", required=False, default=False,
                        help='Enable verbose output')
    parser.add_argument('-a', '--asynchronous', action="store_true", required=False, default=True,
                        help='Use asynchronous inference API')
    parser.add_argument('--streaming', action="store_true", required=False, default=False,
                        help='Use streaming inference API. The flag is only available with gRPC protocol.')
    parser.add_argument('--infinite', action="store_true", required=False, default=False,
                        help='Run at desired throughput indefinitely.')
    parser.add_argument('-m', '--model-name', required=False, type=str, default='densenet_onnx',
                        help='Name of model. Default is densenet_onnx')
    parser.add_argument('--model-version', type=int, required=False, default=-1,
                        help='The version of the model to query. Default is -1.')
    parser.add_argument('-b', '--batch-size', type=int, required=False, default=1,
                        help='Batch size. Default is 1.')
    parser.add_argument('-c', '--classes', type=int, required=False, default=1,
                        help='Number of class results to report. Default is 1.')
    parser.add_argument('--attempts', type=int, required=False, default=1,
                        help='Number of attempts to estimate the max throughput. Default is 1.')
    parser.add_argument('-s', '--scaling', type=str, choices=['NONE', 'INCEPTION', 'VGG'],
                        required=False, default='NONE',
                        help='Type of scaling to apply to image pixels. Default is NONE.')
    parser.add_argument('-u', '--url', type=str, required=False, default='localhost',
                        help='Inference server URL. Default is localhost.')
    parser.add_argument('-p', '--port', type=int, required=False, default=8001,
                        help='Inference server Port. Default is 8001.')
    parser.add_argument('--percentile', type=int, required=False, default=95,
                        help='The SLA percentile. Default is 95.')
    parser.add_argument('--sla', type=float, required=False, default=1.0,
                        help='The SLA. Default is 1 second.')
    parser.add_argument('-f', '--filepath', type=str, required=False,
                        default='Dataset/',
                        help='The filepath of the image files to send to the server. Default is /app/Dataset/.')
    parser.add_argument('-t', '--threads', type=int, required=False, default=(cpu_count() * 24),
                        help='Number of threads. Default is number of cpus times 16.')
    parser.add_argument('-q', '--qps', type=float, required=False, default=5.0,
                        help='Number of queries to send per second. Default is 5.0.')
    parser.add_argument('--qps-delta', type=float, required=False, default=0.0,
                        help='The number of queries to send per second to increment the test by. Default is 0.5.')
    parser.add_argument('-i', '--protocol', type=str, required=False, default='gRPC',
                        help='Protocol (HTTP/gRPC) used to communicate with inference service. Default is gRPC.')

    # Create the benchmark runner with the arguments
    model_benchmark_runner = ModelBenchmarkRunner(parser.parse_args())

    # Run the steps required before running the benchmark
    model_benchmark_runner.setup()

    # Perform the benchmark
    model_benchmark_runner.run_benchmark()
