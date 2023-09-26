import os
import wave
from tqdm import tqdm
from pydub import AudioSegment


pcm_folder = "./preliminary/data/task1"

for filename in tqdm(os.listdir(pcm_folder)):
    if filename.endswith(".pcm"):
        pcm_file_path = os.path.join(pcm_folder, filename)
        wav_file_path = os.path.join(pcm_folder, os.path.splitext(filename)[0] + ".wav")
        
        pcm_audio = AudioSegment.from_file(pcm_file_path, format="raw", frame_rate=44100, channels=2, sample_width=2)
        
        pcm_audio.export(wav_file_path, format="wav")
        
        # os.remove(pcm_file_path)