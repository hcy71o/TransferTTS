import os
import json
import math
import torch
from torch import nn
from torch.nn import functional as F
import librosa
import argparse

import commons
import utils
from data_utils import TextAudioLoader, TextAudioCollate, TextAudioSpeakerLoader, TextAudioSpeakerCollate
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence
from mel_processing import spectrogram_torch, spec_to_mel_torch

from scipy.io.wavfile import write


def get_text(text, hps):
    text_norm = text_to_sequence(text, [])
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

def main(args):

    hps = utils.get_hparams_from_file(args.config)

    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        # n_speakers=hps.data.n_speakers, #* Few-shot
        n_speakers=0, #* Zero-shot
        **hps.model).cuda()
    
    _ = net_g.eval()
    _ = utils.load_checkpoint(args.checkpoint_path, net_g, None)
    
    audio, _ = librosa.load(args.ref_audio, sr=hps.data.sampling_rate)
    audio = torch.from_numpy(audio)
    audio = audio.unsqueeze(0)
    spec = spectrogram_torch(audio, hps.data.filter_length,
    hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length,
    center=False)
    
    spec = torch.squeeze(spec, 0)
    mel = spec_to_mel_torch(
        spec, 
        hps.data.filter_length, 
        hps.data.n_mel_channels, 
        hps.data.sampling_rate,
        hps.data.mel_fmin, 
        hps.data.mel_fmax)
    
    os.makedirs(args.save_path, exist_ok=True)
    
    stn_tst = get_text(args.text, hps)
    
    with torch.no_grad():
        x_tst = stn_tst.cuda().unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
        sid = torch.LongTensor([4]).cuda()
        mel = mel.cuda()
        audio_gen = net_g.infer(x_tst, x_tst_lengths, mel.unsqueeze(0), sid=None, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
        output_file = os.path.join(args.save_path, '{}.wav'.format(args.text[:10]))
        ref_file = os.path.join(args.save_path, 'ref_of_{}.wav'.format(args.text[:10]))
       
        write(output_file, hps.data.sampling_rate, audio_gen)
        write(ref_file, hps.data.sampling_rate, audio[0].cpu().float().numpy())
            
        # audio = y_g_hat.squeeze()
        # audio = audio * MAX_WAV_VALUE
        # audio = audio.cpu().numpy().astype('int16')
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str,
        default="logs/libritts_base/G_319000.pth")
    parser.add_argument('--config', default='configs/libritts.json')
    parser.add_argument("--save_path", type=str, default='wav_results/')
    parser.add_argument("--ref_audio", type=str, required=True,
        help="path to an reference speech audio sample")
    parser.add_argument("--text", type=str,
        help="raw text to synthesize", default = 'in being comparatively modern.')
    
    args = parser.parse_args()
    
    main(args)