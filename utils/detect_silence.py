import os
import librosa
import numpy as np
from tqdm import tqdm
import json

def detect_audio_boundaries(audio_path, threshold=0.1, min_silence_duration=4):
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

if __name__ == "__main__":
    folder_path = './preliminary/data/wav'
    output_path = './preliminary/output/Q3.json'
    results = {}
    
    failed_list = []
    for filename in tqdm(os.listdir(folder_path)):
        if filename.endswith('.wav') and filename.startswith('task3'):
            file_path = os.path.join(folder_path, filename)
            boundaries = detect_audio_boundaries(file_path)
            results[filename.replace('wav','pcm')] = boundaries
            if boundaries == []:
                failed_list.append(filename)
            # output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.json")
    with open(output_path, 'w') as json_file:
        results = dict(sorted(results.items()))
        json.dump(results, json_file, indent=4)

    print(len(failed_list))