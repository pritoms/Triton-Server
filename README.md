# Triton Server Integration with **tensor2tensor**

Triton is a language and compiler for parallel programming. It let's us write deep nearal network compute kernels from python for running on GPU with maximum throughput.

![alt text](http://www.machinedlearnings.com/wp-content/uploads/2018/08/3Comen.gif)

Thanks to Google Brain for their [TensorFlow code](https://github.com/tensorflow/tensor2tensor) that we use here.

Triton currently supports a subset of ops supported by TensorFlow.

# Colab Setup

We need to clone the Triton repo and link to the `tensor2tensor` library in that repo.

You will have to execute the following cell on your own Colab runtime or Google Cloud VM to be able to run this demo.

The next cell uses the following environment variables:

```python
$GIT_REPO_NAME = "triton-inference-server"
$GIT_REPO_OWNER = "triton-inference-server"
$GIT_REPO_BRANCH = "master"
```

Triton expects a runtime environment that has been setup for running [Nvidia docker containers](https://github.com/NVIDIA/nvidia-docker). 

The `./get_t2t_data.sh` downloads the [tensor2tensor](https://github.com/tensorflow/tensor2tensor) data required for this tutorial.

See the [triton](https://github.com/goyalzz/triton) README for more details.

You will have to re-run this cell if you restart your runtime.

**Note:** This is a one time setup cell. You will not need to run this cell again on subsequent runs after you have already setup your runtime.

**Note:** This is not a python cell. We are running shell commands to setup the runtime.

**Note:** For some reason, even though I specify the git branch I want to run this notebook from, I still get a warning about `triton` not being installed. Just ignore this warning as it is not relevant for this notebook.

**Note:** If you don't see the options to select `GPU` as your runtime type, please check what version of Colab you are running. I am on version `1.0.0`.

**Note:** You will have to run the `./get_t2t_data.sh` script *after* you have set the runtime type to `GPU` on the Colab runtime.

**Note:** You will have to re-run the `./get_t2t_data.sh` script on every runtime reboot.

**Note:** You can find more details about the `triton` docker image [here](https://hub.docker.com/r/goyalzz/triton/tags)



```python
%%bash

# Get the git commit hash and branch name
export GIT_COMMIT_HASH=`git rev-parse --short HEAD`
export GIT_BRANCH_NAME=`git rev-parse --abbrev-ref HEAD`

# Clone the Triton Repo
git clone https://github.com/${GIT_REPO_OWNER}/${GIT_REPO_NAME}.git
cd ${GIT_REPO_NAME}
git checkout ${GIT_REPO_BRANCH}

# Setup the new python library
cd python && python setup.py install
cd -

# Link to this notebook from the cloned Triton repo
pwd
ln -s $(pwd) ./triton/notebooks/transformer_triton

# create the triton docker image
cd docker/triton-dev && ./docker_build.sh
cd -

# Download the data required to run this tutorial
./get_t2t_data.sh

# You can find more details about the triton docker image here :
# https://hub.docker.com/r/goyalzz/triton/tags

```

# Environment Setup

We need to import the modules from the `tensor2tensor` library.

We will be working with the `transformer` model in the `Machine Translation` problem.

We will use the `WMT English-German 2014` dataset.

The `WMT 2014 English-German dataset` consists of millions of parallel sentences.

```python
%%bash

# Print out the triton env variables that we have setup
echo $TRITON_IMAGE_NAME
echo $TRITON_CONTAINER_NAME

# Print where the triton container is running
docker ps -f name=${TRITON_CONTAINER_NAME}

# Print the path where the triton data was downloaded
echo $TRITON_DATA_DIR

```

We can also check what GPU our docker container is running on by running the following cell.

```python
%%bash

docker container exec ${TRITON_CONTAINER_NAME} nvidia-smi
```

We now setup the `tensor2tensor` problem and model hparams.

```python
%%bash

# Setup the triton environment variables
export TRITON_IMAGE_NAME="triton-dev:${GIT_COMMIT_HASH}"
export TRITON_CONTAINER_NAME="triton-t2t"
export TRITON_DATA_DIR="/tmp/triton-t2t/data"

# Start the triton container
docker run --rm -it \
    --gpus '"device=0"' \
    -v /usr/src/triton/:/workspace/triton/ \
    -v ${TRITON_DATA_DIR}:/workspace/data/ \
    --name ${TRITON_CONTAINER_NAME} \
    ${TRITON_IMAGE_NAME} \
    bash

# Export the PYTHONPATH to include tensor2tensor
export PYTHONPATH="/workspace/tensor2tensor/:$PYTHONPATH"

# Print the tensor2tensor and triton versions
t2t-trainer --version
tritonc --version

# Set the model hyperparameters
MODEL="transformer"
HPARAMS="transformer_tpu=True"

# Setup the problem
PROBLEM="translate_ende_wmt32k"
DATA_DIR="${TRITON_DATA_DIR}/t2t_data/${PROBLEM}"

# Check for the presence of the data
ls ${DATA_DIR}

```

# Training the Model

We now train the model for 100k steps.

For this tutorial, we will just train on the `train-100` dataset.

You can use the `train-all` dataset or increase the train_steps to 20M if you want to train the full model.

```python
%%bash

mkdir -p ${TRITON_DATA_DIR}/triton_model_dir

# Train the model for 100K steps with the WMT 2014 English-German dataset.
t2t-trainer \
  --data_dir=${DATA_DIR} \
  --problem=${PROBLEM} \
  --model=${MODEL} \
  --hparams_set=${HPARAMS} \
  --output_dir=${TRITON_DATA_DIR}/triton_model_dir \
  --train_steps=100000


```

# Exporting the Model for Inference

We now export the model for inference.

```
%%bash

# Export the model for inference
t2t-exporter \
  --model=${MODEL} \
  --hparams_set=${HPARAMS} \
  --problem=${PROBLEM} \
  --data_dir=${DATA_DIR} \
  --output_dir=${TRITON_DATA_DIR}/triton_model_dir \
  --export_dir=${TRITON_DATA_DIR}/triton_infer_model

```

# Compiling the Model

We now compile the model for inference with Triton.

```python
%%bash

# Compile the model for inference with Triton
tritonc --inference_input_dtype=FP32 \
  --inference_output_dtype=FP32 \
  --first_batch_timeout=60000 \
  --model_name='transformer' \
  --model_version=1 \
  --input_node_name='EnDeTransformer/transformer/encoder/transformer_encoder/while/body/_1/while/body/_0/while/body/ParallelMapDataset/tmp_batch-1' \
  --input_node_shape=1:1:32768:1 \
  --output_node_name='EnDeTransformer/transformer/decoder/transformer_decoder/while/body/while/body/_0/while/body/ParallelMapDataset/tmp_batch-1' \
  --output_node_shape=1:1:32768:1 \
  --output_layer_names='EnDeTransformer/transformer/decoder/transformer_decoder/while/body/while/body/_0/while/body/ParallelMapDataset/tmp_batch-1' \
  --savedmodel_directory=${TRITON_DATA_DIR}/triton_infer_model/export/Servo/1


```

# Running Inference with Triton

We now run inference with a Triton server.

The Triton server will use the newly compiled model.

We supply `triton_infer_input.txt` as our input and the server will translate this input to the output `triton_infer_output.txt`.

The input and output files will be saved in `$TRITON_DATA_DIR`.

If for some reason you don't see the output file, restart the runtime.

```python
%%bash


# Set some names for the input and output files
TRITON_INPUT_FILE=${TRITON_DATA_DIR}/triton_infer_input.txt
TRITON_OUTPUT_FILE=${TRITON_DATA_DIR}/triton_infer_output.txt

# Start a Triton server with the model server api
docker container exec ${TRITON_CONTAINER_NAME} \
  tritonserver --model-repository=`pwd`/triton_model_dir/triton_infer_model/export/Servo/1 &

# Sleep for a little while to let the server start
sleep 3s

# Run the inference
docker container exec ${TRITON_CONTAINER_NAME} \
  triton_client \
    --server=localhost:8000 \
    --model-name=transformer \
    --model-version=1 \
    --input_file=${TRITON_INPUT_FILE} \
    --output_file=${TRITON_OUTPUT_FILE}


```

# Reading Inference Output from the Triton Server

We now read the output from the Triton server and print it out.

```python
%%bash

# We will print the first 20 lines of the output file
head -n 20 ${TRITON_OUTPUT_FILE}

```

# Shutting Down the Triton Server

We now shut down the Triton Server.

```python
%%bash

# Shutdown the Triton server
docker container exec ${TRITON_CONTAINER_NAME} killall tritonserver

```

# Check for the Existence of the Output File

We now check whether the output file exists in the `$TRITON_DATA_DIR`.

If for some reason you don't see the output file, restart the runtime.

```python
%%bash

# Check whether the output file exists
ls -l ${TRITON_OUTPUT_FILE}

```

# Shutdown the Triton Container

We now shutdown the `triton` docker container.

```python
%%bash

docker container stop ${TRITON_CONTAINER_NAME}

```

# Wrapping Up

We have covered the following steps:

* Setup the `triton` docker container.
* Download the `tensor2tensor` dataset required for this tutorial.
* Setup the `tensor2tensor` environment variables.
* Train the transformer model.
* Export the model for inference.
* Segment the model into multiple segments for inference.
* Run inference with the model using the Triton server.
* Read the inference output from the inference server.
* Shutdown the Triton Server.
* Check for the existence of the inference output.
* Shutdown the `triton` docker container.

**Note:** The output from the Triton Server is not very meaningful because we are not supplying the proper inputs to the server.

This is just a tutorial to demonstrate the Triton server integration with the tensor2tensor models.

---

 
# References

Here are some useful links that might help you get started with Triton.

*   [Triton Server](https://github.com/triton-inference-server/server)
*   [Triton Python Client](https://github.com/triton-inference-server/client/tree/master/python)
*   [Triton Release Notes](https://docs.nvidia.com/deeplearning/triton-inference-server/release-notes.html)
*   [Triton API](https://docs.nvidia.com/deeplearning/triton-inference-server/master-user-guide/docs/api.html)
*   [Triton Docker](https://docs.nvidia.com/deeplearning/triton-inference-server/master-user-guide/docs/quickstart.html#docker-container)
*   [Triton Performance](https://docs.nvidia.com/deeplearning/triton-inference-server/master-user-guide/docs/performance.html)
*   [Triton Custom Models](https://docs.nvidia.com/deeplearning/triton-inference-server/master-user-guide/docs/custom_models.html)
*   [Triton Installing Python Client](https://docs.nvidia.com/deeplearning/triton-inference-server/master-user-guide/docs/client_install_python.html)
*   [Triton Architecture](https://docs.nvidia.com/deeplearning/triton-inference-server/master-user-guide/docs/architecture.html)
*   [Triton Metrics](https://docs.nvidia.com/deeplearning/triton-inference-server/master-user-guide/docs/metrics.html)
*   [Triton HTTP/REST API](https://docs.nvidia.com/deeplearning/triton-inference-server/master-user-guide/docs/metrics.html#http-rest-api)
*   [Triton Large Models](https://docs.nvidia.com/deeplearning/triton-inference-server/master-user-guide/docs/large_models.html)
*   [Triton Configuring Large Models](https://docs.nvidia.com/deeplearning/triton-inference-server/master-user-guide/docs/large_models.html#configuring-large-models)
*   [Triton Large Models Architecture](https://docs.nvidia.com/deeplearning/triton-inference-server/master-user-guide/docs/large_models.html#architecture)
*   [Triton Custom Backend](https://docs.nvidia.com/deeplearning/triton-inference-server/master-user-guide/docs/custom_backend.html)
*   [Triton Custom Backend Interface](https://docs.nvidia.com/deeplearning/triton-inference-server/master-user-guide/docs/custom_backend.html#custom-backend-interface)
