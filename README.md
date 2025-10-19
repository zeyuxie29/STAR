# :stars: STAR: Speech-to-Audio Generation via Representation Learning
[![arXiv](https://img.shields.io/badge/arXiv-2509.17164-brightgreen.svg?style=flat-square)](https://arxiv.org/abs/2509.17164)
[![githubio](https://img.shields.io/badge/GitHub.io-Audio_Samples-blue?logo=Github&style=flat-square)](https://zeyuxie29.github.io/STAR)
[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space-blue)](https://huggingface.co/spaces/wsntxxn/STAR)
[![Youtube Demo](https://img.shields.io/badge/YouTube-Video_Demo-red?logo=youtube&logoColor=white)](https://www.youtube.com/watch?v=OeXYA5elwRM)

This work presents **STAR**, the first end-to-end speech-to-audio generation framework, designed to enhance efficiency and address error propagation inherent in cascaded systems. 
It:
* Recognize the potential of the speech-to-audio generation task and have designed the first E2E system STAR;
* Validate E2E STA feasibility via representation learning experiments, showing that spoken sound event semantics can be directly extracted;
* Achieve effective speech-to-audio modal alignment through a bridge network mapping mechanism and a two-stage training strategy;
* Significantly reduces speech processing latency from 156ms to 36ms(≈ 76.9% reduction), whilesurpassing the generation performance of cascaded systems.

### Table of Contents
 - [Data Preparation](#DataPreparation)
 - [Stage1: Bridge Network](#Bridge)
 - [Stage2: STA Generation](#STA)

***

<a id="DataPreparation"></a>
### :scissors: Data Preparation
Generating corresponding speech from captions in Audiocaps, followed by feature extraction using different speech encoders (DAC, Hubert, WavLM):
```shell
git clone https://github.com/zeyuxie29/STAR
python src/data_preparation/vits/vits_inference.py
python src/data_preparation/data_preparation/speech_encoder/hubert_extract_feature.py
```

### :bulb: Stage1: Bridge Network
Pre-train the Bridge Network using sound event labels from AudioSet
```shell
python src/bridge_network/qformer_predictions.py
```

### :seedling: Stage2: STA Generation
Train end-to-end speech-to-audio generation using speech-audio data
 ```shell
sh src/sta_generation/bash_scripts_star/train_star_fm.sh
sh src/sta_generation/bash_scripts_star/infer_multi_gpu
python src/sta_generation/evaluation/star.py --gen_audio_dir {generated_audio_folder}
```

## Acknowledgement
Our code referred to the [WavLM](https://github.com/microsoft/unilm/tree/master/wavlm), [fairseq](https://github.com/facebookresearch/fairseq), [DAC](https://github.com/descriptinc/descript-audio-codec), [SECap](https://github.com/thuhcsi/SECap), [HEAR](https://hearbenchmark.com/). We appreciate their open-sourcing of their code.


