#!nova: pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime
from distutils.core import setup

setup(
    name='kospeech_nova',
    version='latest',
    install_requires=[
        # 'torch==1.7.0',
        # 'levenshtein',
        'librosa >= 0.7.0',
        'numpy==1.21',
        'pandas',
        'tqdm==4.62.3',
        'matplotlib',
        'astropy',
        'sentencepiece',
        'torchaudio==0.6.0',
        'pydub',
        'glob2',
        'datasets'
    ],
)

