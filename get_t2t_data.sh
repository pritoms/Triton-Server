# Debug mode
set -x

# Export the tensor2tensor data directory
export T2T_DATA_DIR=/tmp/triton-t2t/data/t2t_data

# Download the tensor2tensor data required for this tutorial
t2t-datagen \
  --data_dir=${T2T_DATA_DIR} \
  --tmp_dir=/tmp/triton-t2t/data/tmp \
  --problem=translate_ende_wmt32k \
  --t2t_usr_dir=./third_party/tensor2tensor/tensor2tensor/data_generators
