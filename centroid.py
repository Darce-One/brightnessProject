import essentia
import essentia.standard as ess
import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft

def main():
    # input sounds
    x, fs = librosa.load('sounds/kick_drum.wav', mono=True, sr=44100)

    time_in_sec = 2.0
    time_in_samples = int(fs * time_in_sec)

    time_axis = np.arange(time_in_samples) / fs
    sinetone = 0.8 * np.sin(2 * np.pi * 1000 * time_axis)

    # get spectral centroid
    # Spectrum_model = ess.Spectrum(size=N)
    # windowing = ess.Windowing(type='blackmanharris62', zeroPadding=N)
    # Centroid_model = ess.Centroid()
    # spectrum = Spectrum_model(windowing(sinetone))
    # centroid = Centroid_model(spectrum)

    # centroid = librosa.feature.spectral_centroid(y=x, sr=fs, n_fft=x.size)
    mX = np.abs(fft(x))
    centroid = librosa.feature.spectral_centroid(y=sinetone, sr=fs)

    print(centroid)
    plt.plot((np.arange(N + 1) * fs / N), spectrum)
    plt.show()

    # apply bandpass

if __name__ == "__main__":
    main()
