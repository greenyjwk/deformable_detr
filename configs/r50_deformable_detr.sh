#!/usr/bin/env bash

set -x

EXP_DIR=exps/r50_deformable_detr_tensorboard_val
PY_ARGS=${@:1}

MASTER_PORT=29502

python -u main.py \
    --output_dir ${EXP_DIR} \
    ${PY_ARGS}
