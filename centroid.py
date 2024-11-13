import essentia
import essentia.standard as ess
import librosa
import numpy as np
from scipy.fftpack import fft
import soundfile as sf
import pyloudnorm as pyln
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
    parser.add_argument('-o', '--offset_ratio', type=float, default=1.0, help="shifts the target centroid proportionally\
        to the centroid frequency")
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


def apply_bandpass(audio_array: np.ndarray, band_width: float, fs: int, offset_ratio: float = 1, verbose: bool=True) -> np.ndarray:
    # find the initial centroid - this will need to be the centroid after bandpassing!
    target_centroid = get_centroid_avg(audio_array, fs) * offset_ratio

    # create the bandpass filter and apply it initially
    bp_instance = ess.BandPass(bandwidth=band_width, cutoffFrequency=target_centroid, sampleRate=fs)
    bp_audio = bp_instance(audio_array)

    # get the actual centroid after bandpass
    actual_centroid = get_centroid_avg(bp_audio, fs)
    if verbose:
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
        if verbose:
            print(f"intermediary centroid: {actual_centroid}")

        if iter > 29:
            raise TimeoutError("Couldn't converge the bandpass to centroid")

    if verbose:
        print(f'bp_centroid: {actual_centroid}')
    return bp_audio


def apply_bandpass_binarysearch(audio_array: np.ndarray, band_width: float, fs: int, offset_ratio: float = 1, verbose: bool=True) -> np.ndarray:
    target_centroid = get_centroid_avg(audio_array, fs) * offset_ratio

    lower_bound = 0
    upper_bound = fs/2
    cutoff = (lower_bound + upper_bound) / 2

    bp_instance = ess.BandPass(bandwidth=band_width, cutoffFrequency=cutoff, sampleRate=fs)
    bp_audio = bp_instance(audio_array)

    actual_centroid = get_centroid_avg(bp_audio, fs)
    while abs(target_centroid - actual_centroid) > 1:
        if actual_centroid > target_centroid:
            upper_bound = cutoff
        else:
            lower_bound = cutoff
        cutoff = (lower_bound + upper_bound) / 2
        bp_instance = ess.BandPass(bandwidth=band_width, cutoffFrequency=cutoff, sampleRate=fs)
        bp_audio = bp_instance(audio_array)
        actual_centroid = get_centroid_avg(bp_audio, fs)
        if verbose:
            print(f"target: {target_centroid}, actual: {actual_centroid}")
            print(f"lower: {lower_bound}, upper: {upper_bound}, cutoff: {cutoff}")

    return bp_audio


def get_mag_spec(audio_array: np.ndarray) -> np.ndarray:
    return np.abs(fft(audio_array))


def normalise_LUFS(audio_in: np.ndarray, target_lufs: float, fs: float) -> np.ndarray:
    meter = pyln.Meter(fs) # create BS.1770 meter
    loudness_in_lufs = meter.integrated_loudness(audio_in) # measure loudness
    db_diff = target_lufs - loudness_in_lufs
    amplitude_gain = 10 ** (db_diff/20)
    # print(f"db_diff: {db_diff}, db amplitude gain: {amplitude_gain}")
    normalised_audio = audio_in * amplitude_gain
    clips = np.max(np.abs(normalised_audio)) > 1.0
    if clips:
        raise AttributeError("Audio clips")
    return normalised_audio

def calculate_spectral_variance(audio_array: np.ndarray, spectral_centroid: float, fs:float) -> float:
    mX_full = get_mag_spec(audio_array)
    N = mX_full.size
    mX = mX_full[:int(N/2 + 1)]
    freqs = np.arange(N)[:int(N/2 + 1)] * fs / N
    variance = np.sqrt(np.sum(((freqs - spectral_centroid)**2) * mX) / np.sum(mX))
    return variance

def calculate_Lower_upper_spectral_variance(audio_array: np.ndarray, spectral_centroid: float, fs:float) -> tuple[float, float]:
    mX_full = get_mag_spec(audio_array)
    N = mX_full.size
    mX = mX_full[:int(N/2 + 1)]
    freqs = np.arange(N)[:int(N/2 + 1)] * fs / N
    lower_freqs = freqs[freqs < spectral_centroid]
    lower_variance = np.sqrt(np.sum(((lower_freqs - spectral_centroid)**2) * mX[:lower_freqs.size]) / np.sum(mX[:lower_freqs.size]))
    higher_variance = np.sqrt(np.sum(((freqs[freqs > spectral_centroid] - spectral_centroid)**2) * mX[lower_freqs.size:]) / np.sum(mX[lower_freqs.size:]))
    return lower_variance, higher_variance

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
    audio_array, _ = librosa.load(f'sounds/input/{file_name}', mono=True, sr=fs)

    # get target_centroid
    centroid_avg = get_centroid_avg(audio_array, fs)
    variance = calculate_spectral_variance(audio_array, centroid_avg, fs)
    lower_variance_before, higher_variance_before = calculate_Lower_upper_spectral_variance(audio_array, centroid_avg, fs)
    print(f'centroid average = {centroid_avg}')
    print(f"initial variance: {variance}")

    # apply bandpass
    bandwidth = get_bandwidth(args, centroid_avg)
    bp_audio = apply_bandpass(audio_array, band_width=bandwidth, fs=fs, offset_ratio=args.offset_ratio, verbose=False)
    normalised_audio = normalise_LUFS(bp_audio, -25, fs)
    centroid_avg = get_centroid_avg(normalised_audio, fs)
    variance = calculate_spectral_variance(normalised_audio, centroid_avg, fs)
    lower_variance_after, higher_variance_after = calculate_Lower_upper_spectral_variance(normalised_audio, centroid_avg, fs)
    print(f'centroid average = {centroid_avg}')
    print(f"output variance: {variance}")
    print(f"lower variance ratio: {lower_variance_after/lower_variance_before}")
    print(f"higher variance ratio: {higher_variance_after/higher_variance_before}")
    sf.write(f'sounds/output/bp_w_{int(bandwidth)}_{file_name}', normalised_audio, fs)


if __name__ == "__main__":
    main()
