#!/usr/bin/env bash

set -x

EXP_DIR=exps/r50_deformable_detr
PY_ARGS=${@:1}

MASTER_PORT=29501

python -u main.py \
    --output_dir ${EXP_DIR} \
    ${PY_ARGS}
