import torch
import torchaudio
import fairseq
from tqdm import tqdm
import json
import h5py

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
ckpt_path = "~/.cache/audio_tokenizer/hubert_large_ll60k.pt"
models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
model = models[0].eval().to(device)


split_file_path = [
    "data/audiocaps/test/text.json",
    "data/audiocaps/val/text.json",
    "data/audiocaps/train/text.json",   
    ]
samplerate = 22000
for split_file in split_file_path:
    split = split_file.split('/')[-2]
    save_path = f"data/hdf5/hubert_{split}.h5"
    data_json = json.load(open(split_file, 'r'))['audios']

    with h5py.File(save_path, "w") as store:
    
        for item in tqdm(data_json):
            waveform, sample_rate = torchaudio.load(f"data/vits_output/{split}/{item['audio_id']}.wav")
            if sample_rate != 16000:
                waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            with torch.no_grad():
                features, _ = model.extract_features(waveform.to(device))

            store[item['audio_id']] = features.cpu().detach().numpy()

               