import os
import numpy as np
import torchaudio

import torch
import torch.nn as nn
from torch import Tensor

from modules.vocab import KoreanSpeechVocabulary
from modules.data import load_audio, noise_filter
from modules.model import DeepSpeech2


def parse_audio(config, audio_path: str, del_silence: bool = False, audio_extension: str = 'pcm') -> Tensor:
    signal = load_audio(audio_path, del_silence, extension=audio_extension)
    if config.mode != 'train':
        signal = noise_filter(signal)
    feature = torchaudio.compliance.kaldi.fbank(
        waveform=Tensor(signal).unsqueeze(0),
        num_mel_bins=80,
        frame_length=20,
        frame_shift=10,
        window_type='hamming'
    ).transpose(0, 1).numpy()

    feature -= feature.mean()
    feature /= np.std(feature)
    return torch.FloatTensor(feature).transpose(0, 1)


def single_infer(config, model, audio_path):
    device = 'cuda'
    feature = parse_audio(config, audio_path, del_silence=True)
    input_length = torch.LongTensor([len(feature)])
    vocab = KoreanSpeechVocabulary(os.path.join(os.getcwd(), 'labels.csv'), output_unit='character')

    if isinstance(model, nn.DataParallel):
        model = model.module
    model.eval()

    model.device = device
    y_hats = model.recognize(feature.unsqueeze(0).to(device), input_length)
    sentence = vocab.label_to_string(y_hats.cpu().detach().numpy(), target=False)

    return sentence