#! /bin/bash
NUM_WORKERS=1
PORT=8804
MODEL_PATH="/Users/dan/dev/torch-models/llama2-13b-orca-8k-3319"
MODEL_NAME="oasst_llama"

MODEL_NAME=$MODEL_NAME MODEL=$MODEL_PATH uvicorn src.api:app --log-level info \
   --host 0.0.0.0 --port ${PORT} \
   --workers ${NUM_WORKERS}
