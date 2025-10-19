import torch
import torchaudio
import sys

from tqdm import tqdm
import json
import h5py
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

import dac
from audiotools import AudioSignal
import librosa
import torch 
import soundfile as sf
import numpy as np

import copy

def from_latents(quantizer, latents: torch.Tensor):
    """Given the unquantized latents, reconstruct the
    continuous representation after quantization.

    Parameters
    ----------
    latents : Tensor[B x N x T]
        Continuous representation of input after projection

    Returns
    -------
    Tensor[B x D x T]
        Quantized representation of full-projected space
    Tensor[B x D x T]
        Quantized representation of latent space
    """
    z_q = 0
    z_p = []
    codes = []
    dims = np.cumsum([0] + [q.codebook_dim for q in quantizer.quantizers])

    n_codebooks = np.where(dims <= latents.shape[1])[0].max(axis=0, keepdims=True)[
        0
    ]
    for i in range(n_codebooks):
        j, k = dims[i], dims[i + 1]
        print(j, k)
        z_p_i, codes_i = quantizer.quantizers[i].decode_latents(latents[:, j:k, :])
        z_p.append(z_p_i)
        codes.append(codes_i)

        z_q_i = quantizer.quantizers[i].out_proj(z_p_i)
        z_q = z_q + z_q_i

    return z_q, torch.cat(z_p, dim=1), torch.stack(codes, dim=1)

model_path = "~/.cache/descript/dac/weights_24khz_8kbps_0.0.4.pth"

with torch.no_grad():
    model = dac.DAC.load(model_path)
    model.to('cuda')
    sr = 24000

    split_file_path = [
        "data/audiocaps/test/text.json",
        "data/audiocaps/val/text.json",
        "data/audiocaps/train/text.json",   
    ]
    for split_file in split_file_path:
        split = split_file.split('/')[-2]
        save_path = f"data/hdf5/dac_{split}.h5"
        data_json = json.load(open(split_file, 'r'))['audios']

        with h5py.File(save_path, "w") as store:
        
            for item in tqdm(data_json, ncols=60):
                signal, _ = librosa.load(f"data/vits_output/{split}/{item['audio_id']}.wav", sr=sr)
        
            
                signal = torch.tensor(signal)[None, None, :]
                signal = signal.to(model.device)
                x = model.preprocess(signal, sr)
                z, codes, latents, _, _ = model.encode(x)
                z = z.transpose(1, 2)
                store[item['audio_id']] = z.cpu().detach().numpy()




