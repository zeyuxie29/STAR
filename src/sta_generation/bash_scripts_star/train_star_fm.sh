#!/bin/bash

export HF_HOME="~/.hf_cache"
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
accelerate launch --config_file configs/accelerate/nvidia/1gpu.yaml \
 train.py \
 exp_name=audiocaps_star_ft \
 model=single_task_flow_matching_star \
 data@data_dict=star_audiocaps \
 optimizer.lr=2.5e-5 \
 train_dataloader.collate_fn.pad_keys='["waveform", "duration"]' \
 train_dataloader.batch_size=16 \
 val_dataloader.collate_fn.pad_keys='["waveform", "duration"]' \
 warmup_params.warmup_steps=1000 \
 epoch_length=Null \
 epochs=100 \
 +model.pretrained_ckpt="experiments/audiocaps_fm/checkpoints/epoch_100/model.safetensors" \
 +trainer.resume_from_checkpoint='experiments/audiocaps_star_ft/checkpoints/epoch_92' \