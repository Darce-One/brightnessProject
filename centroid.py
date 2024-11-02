import essentia
import essentia.standard as ess
import librosa
import numpy as np
from scipy.fftpack import fft
import soundfile as sf
from argparse import ArgumentParser, Namespace
# import matplotlib.pyplot as plt


def parse_arguments() -> Namespace:
    # Define arguments
    parser = ArgumentParser(description="This script bandpasses an input audio file \
        while keeping the spectral centroid intact. Please specify either a bandwidth or a \
        bandwidth ratio using -w or -p")
    parser.add_argument('-w', '--width', type=float,
        help="Sets the bandwidth of the bandpass filter", required=False)
    parser.add_argument('-r', '--width_ratio', type=float, required=False)
    parser.add_argument('-f', '--file_name', type=str, required=True, help="Specifies the audio file to \
        apply the filter to")
    parser.add_argument('-s', '--sample_rate', type=int, default=44100, help="Sets the sample rate")

    # Parse args
    args = parser.parse_args()

    if args.width == None and args.width_ratio == None:
        raise TypeError("Use either -w or -r to specify a bandwidth or bandwith ratio")

    if not (args.width == None) and not (args.width_ratio == None):
        raise TypeError("Please specify either -w or -r, not both")

    return args


def get_bandwidth(args: Namespace, target_centroid: float) -> float:
    if not args.width == None:
        return args.width
    return args.width_ratio * target_centroid


def get_centroid_avg(audio_array: np.ndarray, fs: int) -> float:
    centroid = librosa.feature.spectral_centroid(y=audio_array, sr=fs)
    avg_centroid = centroid.mean()
    return avg_centroid


def apply_bandpass(audio_array: np.ndarray, band_width: float, fs: int) -> np.ndarray:
    # find the initial centroid - this will need to be the centroid after bandpassing!
    target_centroid = get_centroid_avg(audio_array, fs)

    # create the bandpass filter and apply it initially
    bp_instance = ess.BandPass(bandwidth=band_width, cutoffFrequency=target_centroid, sampleRate=fs)
    bp_audio = bp_instance(audio_array)

    # get the actual centroid after bandpass
    actual_centroid = get_centroid_avg(bp_audio, fs)
    print(f"intermediary centroid: {actual_centroid}")

    # set up gradient descent
    test_centroid = target_centroid
    iter = 0
    while abs(target_centroid - actual_centroid) > 0.2 and iter < 30:
        # gradient descent on test_centoid using abs(target_centroid - actual_centroid) as the cost function
        test_centroid = test_centroid + (target_centroid - actual_centroid)
        assert test_centroid > 0, "Failed to filter the sound correctly"
        bp_instance = ess.BandPass(bandwidth=band_width, cutoffFrequency=test_centroid, sampleRate=fs)
        bp_audio = bp_instance(audio_array)
        actual_centroid = get_centroid_avg(bp_audio, fs)
        iter += 1
        print(f"intermediary centroid: {actual_centroid}")

    print(f'bp_centroid: {actual_centroid}')
    return bp_audio


def get_mag_spec(audio_array: np.ndarray) -> np.ndarray:
    return np.abs(fft(audio_array))


def synthesise_sine(time_in_sec: float, fs: int, frequency: float, amplitude: float) -> np.ndarray:
    time_in_samples = int(fs * time_in_sec)
    time_axis = np.arange(time_in_samples) / fs
    sinetone = amplitude * np.sin(2 * np.pi * frequency * time_axis)
    return sinetone

#======#
# MAIN #
# =====#

def main():
    # get arguments
    args = parse_arguments()
    fs = args.sample_rate
    file_name = args.file_name

    # input sounds
    audio_array, _ = librosa.load(f'sounds/{file_name}', mono=True, sr=fs)

    # get target_centroid
    centroid_avg = get_centroid_avg(audio_array, fs)
    print(f'centroid average = {centroid_avg}')

    # apply bandpass
    bandwidth = get_bandwidth(args, centroid_avg)
    bp_audio = apply_bandpass(audio_array, band_width=bandwidth, fs=fs)
    sf.write(f'sounds/outs/bp_w_{int(bandwidth)}_{file_name}', bp_audio, fs)


if __name__ == "__main__":
    main()
