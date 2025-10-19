import torch

torch.multiprocessing.set_sharing_strategy('file_system')
import argparse
from collections import defaultdict

import os
import numpy as np
import librosa
from pathlib import Path
from tqdm import tqdm
from copy import deepcopy

# Ref: https://github.com/haoheliu/audioldm_eval/tree/main
# This script uses a locally modified version of audioldm_eval.
from audioldm_eval import EvaluationHelper

# Ref: https://github.com/LAION-AI/CLAP
# The ref command for installing: pip install laion-clap
import laion_clap
from utils.general import read_jsonl_to_mapping, audio_dir_to_mapping

import os
import shutil
from pathlib import Path
from laion_clap.clap_module.factory import load_state_dict as clap_load_state_dict
from path_tta import compute_clap_metrics, AudioTextDataset, evaluate

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ref_audio_dir",
        "-r",
        type=str,
        default="data/audiocaps/audio/test",
        help="path to reference audio jsonl file"
    )
    parser.add_argument(
        "--ref_caption_jsonl",
        "-rc",
        type=str,
        default="data/audiocaps/test/caption.jsonl",
        help="path to reference caption jsonl file"
    )
    parser.add_argument(
        "--gen_audio_dir",
        "-gd",
        type=str,
        help="path to generated audio directory"
    )
    parser.add_argument(
        "--output_file",
        "-o",
        default="",
        help="path to output file"
    )
    parser.add_argument(
        "--task",
        "-t",
        type=str,
        default="sta_test",
    )
    parser.add_argument(
        "--num_workers",
        "-c",
        default=4,
        type=int,
        help="number of workers for parallel processing"
    )
    parser.add_argument(
        "--clap_per_audio",
        "-p",
        action="store_true",
        help="calculate and store CLAP score for each audio clip"
    )
    parser.add_argument(
        "--recalculate",
        action="store_true",
        help="recalculate embeddings for metric scores"
    )

    args = parser.parse_args()

    evaluate(args)
