#!/bin/bash

export HF_HOME="~/.hf_cache"
accelerate launch --config_file configs/accelerate/nvidia/1gpu.yaml \
    inference.py \
    +exp_dir=experiments/audiocaps_star_ft \
    data@data_dict=star_audiocaps \
    infer_args.guidance_scale=5.0 \
    infer_args.num_steps=20 \
    test_dataloader.collate_fn.pad_keys='[]'