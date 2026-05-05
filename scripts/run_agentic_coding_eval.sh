#!/usr/bin/env bash
set -euo pipefail

: "${DATA_PATH:?Set DATA_PATH to the root directory of local agentic-coding tasks}"
: "${OUTPUT_PATH:?Set OUTPUT_PATH to the output directory}"
: "${MODEL_TYPE:?Set MODEL_TYPE to 'lexical' or 'sentence-transformer'}"

TASK_ARGS=""
if [ -n "${TASK_NAME:-}" ]; then
    TASK_ARGS="--task-name ${TASK_NAME}"
fi

MODEL_ARGS=""
if [ -n "${MODEL_NAME:-}" ]; then
    MODEL_ARGS="--model-name ${MODEL_NAME}"
fi

TOP_K="${TOP_K:-100}"

python -m rteb.evaluate_agentic_coding \
    --data-path "${DATA_PATH}" \
    --output "${OUTPUT_PATH}" \
    --model-type "${MODEL_TYPE}" \
    --top-k "${TOP_K}" \
    ${TASK_ARGS} \
    ${MODEL_ARGS}
