#!/bin/bash

py_sc="python train_edge_model.py \
        --config configs/config_edge_model_hpc.yaml"

gpu_new_ "$py_sc"