import argparse
import json
import pydub
from pydub.exceptions import CouldntDecodeError
import tqdm

def arg_parse():
    parser = argparse.ArgumentParser(description='Korean SR Contest 2023')
    parser.add_argument(
        'audiolist', 
        type=str, 
        help="Path to .txt file where target files are listed")
    parser.add_argument(
        'outfile', type=str, 
        help="Path to .json file where the output is to be written")

    args = parser.parse_args()

    return args


'''
    - file_list : audio file list (wavlist.txt)
    - out_file : output file (Q2.json)
'''

def detect_wav_error(file_list, out_file):
    error_files = list()
    with open(file_list, 'r') as f:
        for line in tqdm.tqdm(f):
            file = line.strip()
            try:
                audio = pydub.AudioSegment.from_wav(file)
                if not len(audio): #header-only
                    error_files.append(file.replace("../", "").replace("./", ""))
                
                if audio.max_dBFS >= 0: #clipping error
                    error_files.append(file.replace("../", "").replace("./", ""))
                
            except CouldntDecodeError: #data-only
                error_files.append(file.replace("../", "").replace("./", ""))
        
            except Exception as e: # any other exception
                error_files.append(file.replace("../", "").replace("./", ""))
    
    with open(out_file, 'w', encoding='utf-8') as jsonf:
        json.dump({"error_list": error_files}, jsonf, indent=4, ensure_ascii=False)


def main():
    args = arg_parse()

    # Q2
    detect_wav_error(args.audiolist, args.outfile)


if __name__ == "__main__":
    main()