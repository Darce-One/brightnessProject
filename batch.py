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

# def batch_run(config: dict):
#     files = os.listdir(config['input_folder'])
#     fs = config['sample_rate']
#     for file in files:

#         audio, _ = librosa.load(config['input_folder'] + file, sr=fs)
#         filename = file.split('.')[0]
#         if config['normalisation']:
#             out_audio = centroid.normalise_LUFS(audio, config['target'], fs)
#             sf.write(config['output_folder'] + filename + "_full.wav", out_audio, fs)
#         else:
#             sf.write(config['output_folder'] + filename + "_full.wav", audio, fs)

#         for bw in config['bandwidths']:
#             try:
#                 print(f"Processing file {filename} with bandwidth {bw}...")
#                 out_audio = centroid.apply_bandpass(audio, bw, fs, verbose=False)
#             except TimeoutError as err:
#                 print(f"{err}. \n{filename} with bandwidth: {bw}")
#                 if config['normalisation']:
#                     out_audio = centroid.normalise_LUFS(out_audio, config['target'], fs)
#                 sf.write(config['output_folder'] + filename + '_' + str(bw) + '.wav', out_audio, fs)
#             except (TimeoutError, AttributeError) as err:
#                 print(f"{err}. \n{filename} with bandwidth: {bw}")

def batch_run(config: dict):
    files = [file for file in os.listdir(config['input_folder']) if file.endswith('.wav')]
    fs = config['sample_rate']

    for file in files:
        # Load the audio file
        audio, _ = librosa.load(config['input_folder'] + file, sr=fs)
        filename = file.split('.')[0]

        print(f"Processing file {filename} with full bandwidth...")
        # Full file normalization and writing
        if config['normalisation']:
            out_audio = centroid.normalise_LUFS(audio, config['target'], fs)
            sf.write(config['output_folder'] + filename + "_full.wav", out_audio, fs)
        else:
            sf.write(config['output_folder'] + filename + "_full.wav", audio, fs)

        # Process each bandwidth
        for bw in config['bandwidths']:
            try:
                print(f"Processing file {filename} with bandwidth {bw}...")
                out_audio = centroid.apply_bandpass_binarysearch(audio, bw, fs, verbose=False)

                # Apply normalization if needed
                if config['normalisation']:
                    out_audio = centroid.normalise_LUFS(out_audio, config['target'], fs)

                # Write the output audio file
                sf.write(config['output_folder'] + filename + '_' + str(bw) + '.wav', out_audio, fs)

            except (TimeoutError, AttributeError) as err:
                print(f"{err}. \n{filename} with bandwidth: {bw}")


def main():
    args = parse_arguments()
    config = read_config(args)
    batch_run(config)

if __name__ == "__main__":
    main()
