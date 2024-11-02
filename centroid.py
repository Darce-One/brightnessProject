import essentia
import essentia.standard as ess
import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
import soundfile as sf

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
    print(f'actual_centroid: {actual_centroid}')

    # set up gradient descent
    test_centroid = target_centroid
    iter = 0
    # while abs(target_centroid - actual_centroid) > 2 or iter:
    for iter in range(10):
        # gradient descent on test_centoid using abs(target_centroid - actual_centroid) as the cost function
        test_centroid = test_centroid + 0.2 * (target_centroid - actual_centroid) # check that is correct
        bp_instance = ess.BandPass(bandwidth=band_width, cutoffFrequency=test_centroid, sampleRate=fs)
        bp_audio = bp_instance(audio_array)
        actual_centroid = get_centroid_avg(bp_audio, fs)
        print(f'actual_centroid: {actual_centroid}')
    return bp_audio

def get_mag_spec(audio_array: np.ndarray)->np.ndarray:
    return np.abs(fft(audio_array))

def synthesise_sine(time_in_sec: float, fs: int, frequency: float, amplitude: float) -> np.ndarray:
    time_in_samples = int(fs * time_in_sec)
    time_axis = np.arange(time_in_samples) / fs
    sinetone = amplitude * np.sin(2 * np.pi * frequency * time_axis)
    return sinetone

def main():
    # input sounds
    audio_array, fs = librosa.load('sounds/oboe-A4.wav', mono=True, sr=44100)
    fs = int(fs)

    centroid_avg = get_centroid_avg(audio_array, fs)
    print(f'centroid average = {centroid_avg}')

    # apply bandpass
    bandwidth = 200
    bp_audio = apply_bandpass(audio_array, band_width=bandwidth, fs=fs)
    sf.write('sounds/outs/bp_oboe-A4.wav', bp_audio, fs)







if __name__ == "__main__":
    main()
