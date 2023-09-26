'''
    한국어AI 경진대회 2023 제출 코드 입출력 예시
'''

import argparse
import os
import librosa
import numpy as np
import json
from pydub import AudioSegment

def arg_parse():
    parser = argparse.ArgumentParser(description='Korean SR Contest 2023')
    parser.add_argument('--audiolist', type=str)
    parser.add_argument('--outfile', type=str)

    args = parser.parse_args()

    return args


'''
    - file_list : audio file list (pcmlist.txt)
    - out_file : output file (Q3.json)
'''

def convert_pcm_to_wav(input_file):
    output_file = input_file.replace('pcm','wav')

    sample_width = 2  # 16 비트
    frame_rate = 16000  # 16 kHz
    channels = 1  # Mono

    audio = AudioSegment.from_file(input_file, format="raw",
                                   sample_width=sample_width,
                                   frame_rate=frame_rate,
                                   channels=channels)

    audio.export(output_file, format="wav")

def detect_silence(audio_path, threshold=0.1, min_silence_duration=4):
    y, sr = librosa.load(audio_path)
    energy = librosa.feature.rms(y=y)[0]
    normalized_energy = (energy - np.min(energy)) / (np.max(energy) - np.min(energy))
    silence_segments = []
    is_silence = False
    silence_start = None
    for i, e in enumerate(normalized_energy):
        if e < threshold and is_silence == False:
            silence_start = i
            is_silence = True
            
        elif e >= threshold and is_silence == True:
            silence_end = i
            ss_sec = silence_start  * (len(y) / len(energy)) / sr
            se_sec = silence_end * (len(y) / len(energy)) / sr
            silence_duration = se_sec - ss_sec
            if silence_duration >= min_silence_duration:
                silence_segments.append({"beg":ss_sec, "end":se_sec})
            is_silence = False
    return silence_segments


def main():
    args = arg_parse()

    file_list_path = args.audiolist
    output_path = args.outfile

    file_paths = []
    results = {}

    with open(file_list_path, 'r') as file:
        for line in file:
            file_paths.append(line.strip())

    for path in file_paths:
        convert_pcm_to_wav(path)
        # Q2
        filename = os.path.basename(path)
        wav_path = path.replace('pcm', 'wav') 
        boundaries = detect_silence(wav_path)
        results[filename] = boundaries

    with open(output_path, 'w') as json_file:
        results = dict(sorted(results.items()))
        json.dump(results, json_file, indent=4)

if __name__ == "__main__":
    main()