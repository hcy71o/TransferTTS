from multiprocessing.sharedctypes import Value
from text import _clean_text
import numpy as np
import librosa
import os
from pathlib import Path
from scipy.io.wavfile import write
from joblib import Parallel, delayed
from scipy.interpolate import interp1d
import json


def write_single(output_folder, wav_fname, resample_rate, top_db=None):
    data, sample_rate = librosa.load(wav_fname, sr=None)
    # trim audio
    if top_db is not None:
        trimmed, _ = librosa.effects.trim(data, top_db=top_db)
    else:
        trimmed = data
    # resample audio
    resampled = librosa.resample(trimmed, sample_rate, resample_rate)
    y = (resampled * 32767.0).astype(np.int16)
    wav_fname = wav_fname.split('/')[-1]
    target_wav_fname = os.path.join(output_folder, wav_fname)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    write(target_wav_fname, resample_rate, y)

    return y.shape[0] / float(resample_rate)


def prepare_align_and_resample(data_dir, sr):
    wav_foder_names = ['train-clean-100', 'train-clean-360']
    wavs = []
    for wav_folder in wav_foder_names:
        wav_folder = os.path.join(data_dir, wav_folder)
        wav_fname_list = [str(f) for f in list(Path(wav_folder).rglob('*.wav'))]

        output_wavs_folder_name = 'wav{}'.format(sr//1000)
        output_wavs_folder = os.path.join(data_dir, output_wavs_folder_name)
        if not os.path.exists(output_wavs_folder):
            os.mkdir(output_wavs_folder)

        for wav_fname in wav_fname_list:
            _sid = wav_fname.split('/')[-3]
            output_folder = os.path.join(output_wavs_folder, _sid)
            wavs.append((output_folder, wav_fname))

    lengths = Parallel(n_jobs=10, verbose=1)(
        delayed(write_single)(wav[0], wav[1], sr) for wav in wavs
    )