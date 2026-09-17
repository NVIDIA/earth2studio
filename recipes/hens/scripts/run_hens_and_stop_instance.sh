#!/usr/bin/env bash
set -o pipefail

INSTANCE_ID="${INSTANCE_ID:?Set INSTANCE_ID first}"
REGION="${AWS_REGION:-us-east-2}"

HENS_DIR="$HOME/projects/earth2studio/recipes/hens"

cd "$HENS_DIR" || exit 1

export PATH="$HOME/.local/bin:$PATH"
export LD_LIBRARY_PATH="$HENS_DIR/.venv/lib/python3.13/site-packages/nvidia/cudnn/lib:$HENS_DIR/.venv/lib/python3.13/site-packages/nvidia/cublas/lib:${LD_LIBRARY_PATH:-}"

echo "Starting HENS at $(date -u)"

if /usr/bin/time -v \
    uv run python main.py --config-name=beryl_poc.yaml \
    2>&1 | tee run.log
then
    echo "HENS completed successfully at $(date -u)"

    # Optional: copy outputs to S3 before stopping
    aws s3 cp hens_beryl/ \
      s3://hens-beryl/beryl_poc/outputs \
      --recursive \
      --region us-east-2

    echo "Stopping instance: $INSTANCE_ID"

    aws ec2 stop-instances \
      --instance-ids "$INSTANCE_ID" \
      --region "$REGION"
else
    echo "HENS failed at $(date -u)"
    # echo "Instance will remain running for diagnosis."

    echo "Stopping instance: $INSTANCE_ID"
    aws ec2 stop-instances \
      --instance-ids "$INSTANCE_ID" \
      --region "$REGION"
    exit 1
fi
