'''
    한국어AI 경진대회 2023 제출 코드 입출력 예시
'''

import argparse
#import wave
import json
import pydub
from pydub.exceptions import CouldntDecodeError
import tqdm 

def arg_parse():
    parser = argparse.ArgumentParser(description='Korean SR Contest 2023')
    parser.add_argument('audiolist', type=str)
    parser.add_argument('outfile', type=str)

    args = parser.parse_args()

    return args


'''
    - file_list : audio file list (wavlist.txt)
    - out_file : output file (Q2.json)
'''
def detect_wav_error(file_list, out_file):
    #
    # YOUR CODE HRER
    error_files = list()
    with open(file_list, 'r') as f:
        for line in tqdm.tqdm(f):
            file = line.strip()
            try:
                audio = pydub.AudioSegment.from_wav(file)
                if not len(audio): #header-only
                    error_files.append(file.split("/")[-1])
                
                if audio.max_dBFS >= 0: #clipping error
                    error_files.append(file.split("/")[-1])
                
            except CouldntDecodeError: #data-only
                error_files.append(file.split("/")[-1])
        
            except Exception as e:
                error_files.append(file.split("/")[-1])
    
    with open(out_file, 'w') as jsonf:
        json.dump({"error_list": error_files}, jsonf, indent=4)


def main():
    args = arg_parse()

    # Q2
    detect_wav_error(args.audiolist, args.outfile)


if __name__ == "__main__":
    main()