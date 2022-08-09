# Debug mode
set -x

# We need to export the hash of the current repository commit as an env var
export GIT_COMMIT_HASH=`git rev-parse --short HEAD`

# We need to export the branch name as an env var
export GIT_BRANCH_NAME=`git rev-parse --abbrev-ref HEAD`

# Build the triton docker image
docker build -t triton-dev:${GIT_COMMIT_HASH} .

# Create the triton model repository
docker run --rm -it --name triton-t2t -v /tmp/triton-t2t/data:/workspace/data/ -v /tmp/triton-t2t/triton_model_dir:/workspace/triton_model_dir/ triton-dev:${GIT_COMMIT_HASH} mkdir -p /workspace/triton_model_dir
