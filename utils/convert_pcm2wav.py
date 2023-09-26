import os
from pydub import AudioSegment

folder_path = './preliminary/data/'
output_path = './preliminary/data/wav'

if not os.path.exists(output_path):
    os.makedirs(output_path)

def convert_pcm_to_wav(input_file, output_folder):
    file_name = os.path.splitext(os.path.basename(input_file))[0]
    output_file = os.path.join(output_folder, f"{file_name}.wav")

    sample_width = 2  # 16 비트
    frame_rate = 16000  # 16 kHz
    channels = 1  # Mono

    audio = AudioSegment.from_file(input_file, format="raw",
                                   sample_width=sample_width,
                                   frame_rate=frame_rate,
                                   channels=channels)

    audio.export(output_file, format="wav")

for root, dirs, files in os.walk(folder_path):
    for file in files:
        file_path = os.path.join(root, file)

        if file.endswith(".pcm"):
            convert_pcm_to_wav(file_path, output_path)