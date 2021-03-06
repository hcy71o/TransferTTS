# TransferTTS (Zero-shot VITS) - PyTorch Implementation (-Ongoing-)

Unofficial PyTorch Implementation of [Transfer Learning Framework for Low-Resource Text-to-Speech using a Large-Scale Unlabeled Speech Corpus](https://arxiv.org/abs/2203.15447). Most of codes are based on [VITS](https://github.com/jaywalnut310/vits)

0. MelStyleEncoder from [StyleSpeech](https://arxiv.org/abs/2106.03153) is used instead of the reference encoder.
1. Implementation of untranscribed data training is omitted.
2. [LibriTTS]((https://research.google/tools/datasets/libri-tts/)) dataset (train-clean-100 and train-clean-360) is used. Sampling rate is set to 22050Hz.

<p align="center">
    <img src="img/Overview.jpg" width="80%">
</p>

# Branch
- [x] TransferTTS (Zero-shot) (`master` branch)
- [TODO] TransferTTS (Few-shot) (`fewshot` branch)

## Pre-requisites (from VITS)
0. Python >= 3.6
0. Clone this repository
0. Install python requirements. Please refer [requirements.txt](requirements.txt)
    1. You may need to install espeak first: `apt-get install espeak`
0. Build Monotonic Alignment Search and run preprocessing if you use your own datasets.
```sh
# Cython-version Monotonoic Alignment Search
cd monotonic_align
python setup.py build_ext --inplace
```
## Preprocessing
 
Run 
```
python prepare_wav.py --data_path [LibriTTS DATAPATH]
```
for some preparations.

## Training

Train your model with
```
python train_ms.py -c configs/libritts.json -m libritts_base
```

## Inference
```
python inference.py --ref_audio [REF AUDIO PATH] --text [INPUT TEXT]
```

# References
- [TransferTTS](https://arxiv.org/abs/2203.15447)
- [VITS](https://arxiv.org/abs/2106.06103)
- [Meta-StyleSpeech](https://arxiv.org/abs/2106.03153)
