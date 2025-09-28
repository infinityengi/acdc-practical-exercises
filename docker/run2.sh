#!/bin/sh
#!/bin/bash
# Exit on errors
set -e

# # Exit on unhandled errors, but allow commands to fail where we handle failures explicitly
# set -o errexit
# set -o pipefail

DIR="$(cd -P "$(dirname "$0")" && pwd)"
MOUNT_DIR="$(dirname "$DIR")"

# ensure variables are quoted when used
TARGET_UID=1000
CONTAINER_NAME='ika-acdc-notebooks'
IMAGE='rwthika/acdc-notebooks:latest'

# grant temporary write permission to uid 1000
echo "Giving temporary write permission to uid $TARGET_UID on $MOUNT_DIR"
setfacl -R -m u:${TARGET_UID}:rwx "$MOUNT_DIR" || {
  echo "Warning: setfacl failed (you may not have acl installed or lack permissions). Continuing."
}

# ensure ACL is removed on exit (normal, error, or interrupt)
_cleanup() {
  echo "Removing temporary write permission for uid $TARGET_UID on $MOUNT_DIR"
  setfacl -R -x u:${TARGET_UID} "$MOUNT_DIR" || true
}
trap _cleanup EXIT INT TERM

# detect whether Docker can run a GPU-enabled container
echo "Checking whether Docker can access the GPU..."
if docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi >/dev/null 2>&1; then
  GPU_FLAG="--gpus all"
  echo "GPU available to Docker. Launching container with GPU support."
else
  GPU_FLAG=""
  echo "GPU not available to Docker. Launching container without GPU."
fi

# common docker run args
DOCKER_ARGS="
--name=${CONTAINER_NAME}
--rm
--interactive
--tty
--publish 8888:8888
--publish 9090:9090
--volume ${MOUNT_DIR}:/home/jovyan/acdc
"

# run the container
echo "Starting docker container ${CONTAINER_NAME} (image: ${IMAGE})"
# shellcheck disable=SC2086
docker run ${GPU_FLAG} ${DOCKER_ARGS} ${IMAGE}
