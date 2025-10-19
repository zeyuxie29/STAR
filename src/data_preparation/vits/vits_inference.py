import os
import matplotlib.pyplot as plt
import IPython.display as ipd

import json
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import commons
import utils
from data_utils import TextAudioLoader, TextAudioCollate, TextAudioSpeakerLoader, TextAudioSpeakerCollate
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence
import soundfile as sf
from scipy.io.wavfile import write
from tqdm import tqdm


def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

hps = utils.get_hparams_from_file("configs/ljs_base.json")
net_g = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    **hps.model).cuda()
_ = net_g.eval()

_ = utils.load_checkpoint("~/.cache/vits_pretrained/pretrained_ljs.pth", net_g, None)

split_file_path = [
    "data/audiocaps/test/text.json",
    "data/audiocaps/val/text.json",
    "data/audiocaps/train/text.json",
]
samplerate = 22000
for split_file in split_file_path:
    split = split_file.split('/')[-2]
    save_path = f"data/vits_output/{split}"
    os.makedirs(save_path, exist_ok=True)
    data_json = json.load(open(split_file, 'r'))['audios']
    for item in tqdm(data_json):
        output_path = f"{save_path}/{item['audio_id']}.wav"
        if os.path.exists(output_path):
            continue
        stn_tst = get_text(item["captions"][0]["caption"], hps)
        with torch.no_grad():
            x_tst = stn_tst.cuda().unsqueeze(0)
            x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
            audio = net_g.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
        sf.write(output_path, audio, samplerate)

        