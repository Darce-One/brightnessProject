import centroid
from argparse import ArgumentParser, Namespace
import essentia
import essentia.standard as ess
import librosa
import numpy as np
from scipy.fftpack import fft
import soundfile as sf
import yaml
import os

def parse_arguments() -> Namespace:
    parser = ArgumentParser(description="This script runs the yml batch file")
    parser.add_argument('-f', '--file_name', type=str, required=True, help="Specifies the yml file to run")
    args = parser.parse_args()
    return args

def read_config(args: Namespace) -> dict:
    with open(args.file_name, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)
    
def batch_run(config: dict):
    files = os.listdir(config['input_folder'])
    fs = config['sample_rate']
    for file in files:
        
        audio, _ = librosa.load(config['input_folder'] + file, sr=fs)
        filename = file.split('.')[0]
        
        sf.write(config['output_folder'] + filename + "_full.wav", audio, fs)

        for bw in config['bandwidths']:
            bp_audio = centroid.apply_bandpass(audio, bw, fs, verbose=False)
            sf.write(config['output_folder'] + filename + '_' + str(bw) + '.wav', bp_audio, fs)

def main():
    args = parse_arguments()
    config = read_config(args)
    batch_run(config)

if __name__ == "__main__":
    main()

