from tqdm import tqdm
import json
import h5py

import torch
from WavLM import WavLM, WavLMConfig
import torchaudio

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
checkpoint = torch.load('~/.cache/audio_tokenizer/WavLM-Large.pt')
cfg = WavLMConfig(checkpoint['cfg'])
model = WavLM(cfg)
model.load_state_dict(checkpoint['model'])
model.eval().to(device)


split_file_path = [
    "data/audiocaps/test/text.json",
    "data/audiocaps/val/text.json",
    "data/audiocaps/train/text.json",   
    ]
samplerate = 22000
for split_file in split_file_path:
    split = split_file.split('/')[-2]
    save_path = f"data/hdf5/wavlm_{split}.h5"
    data_json = json.load(open(split_file, 'r'))['audios']
    
    with h5py.File(save_path, "w") as store:
        for item in tqdm(data_json):
            waveform, sample_rate = torchaudio.load(f"data/vits_output/{split}/{item['audio_id']}.wav")
            if sample_rate != 16000:
                waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            if cfg.normalize:
                waveform = torch.nn.functional.layer_norm(waveform , waveform.shape)
            with torch.no_grad():
                rep = model.extract_features(waveform.to(device))[0]
            store[item['audio_id']] = rep.cpu().detach().numpy()



