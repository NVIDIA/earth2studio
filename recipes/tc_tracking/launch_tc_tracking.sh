#!/bin/bash
# Launch the TC tracking container on the local machine.
#
# The earth2studio repository is mounted at /workspace, the shell starts in the
# recipe directory. The container runs as the calling user, so everything it
# writes (outputs, caches) is owned by that user on the host. Model weights and
# data are cached on the host and survive container restarts.
#
#   ./launch_tc_tracking.sh                       # interactive shell
#   ./launch_tc_tracking.sh python main.py --config-name=helene.yaml
#
# Settings can be overridden through environment variables, e.g.
#   GPUS=0,1 IMAGE=tc_tracking:1.0-aifs2 ./launch_tc_tracking.sh
# LOCAL_E2S=1 uses the earth2studio package of the mounted repository instead of
# the one installed in the image, for developing recipe and library together.
#
# DETACH=1 starts the container in the background instead, to run several jobs
# in it, e.g.
#   DETACH=1 ./launch_tc_tracking.sh
#   docker exec tc_tracking python main.py --config-name=helene.yaml
#   docker stop tc_tracking
set -euo pipefail

IMAGE=${IMAGE:-tc_tracking:1.0}
GPUS=${GPUS:-1}                                       # comma-separated device ids
E2S_CACHE=${E2S_CACHE:-${HOME}/.cache/earth2studio}   # model weights and data
HF_CACHE=${HF_CACHE:-${HOME}/.cache/huggingface}      # Hugging Face hub downloads
CDSAPIRC=${CDSAPIRC:-${HOME}/.cdsapirc}               # credentials for CDS_ERA5
LOCAL_E2S=${LOCAL_E2S:-0}
DETACH=${DETACH:-0}
NAME=${NAME:-tc_tracking}                             # container name with DETACH=1

REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
mkdir -p "${E2S_CACHE}" "${HF_CACHE}"

MOUNTS=(
    -v "${REPO_DIR}:/workspace"
    -v "${E2S_CACHE}:/cache/earth2studio"
    -v "${HF_CACHE}:/cache/huggingface"
)
if [[ -f "${CDSAPIRC}" ]]; then
    MOUNTS+=(-v "${CDSAPIRC}:/cache/cdsapirc:ro" -e CDSAPI_RC=/cache/cdsapirc)
fi
if [[ "${LOCAL_E2S}" == 1 ]]; then
    MOUNTS+=(-e PYTHONPATH=/workspace)
fi

if [[ "${DETACH}" == 1 ]]; then
    RUN_MODE=(-d --name "${NAME}")
    set -- sleep infinity
else
    # interactive terminal only if there is one, so the script also works in batch jobs
    RUN_MODE=()
    [[ -t 0 && -t 1 ]] && RUN_MODE=(-it)
fi

# --ipc=host: TempestExtremes stages its input in /dev/shm, docker's default is 64 MB
docker run --rm "${RUN_MODE[@]}" \
    --gpus "\"device=${GPUS}\"" \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --user "$(id -u):$(id -g)" \
    -e HOME=/home/tc_hunt \
    -e USER="$(id -un)" \
    -e EARTH2STUDIO_CACHE=/cache/earth2studio \
    -e HF_HOME=/cache/huggingface \
    "${MOUNTS[@]}" \
    --workdir /workspace/recipes/tc_tracking \
    "${IMAGE}" \
    "${@:-bash}"
