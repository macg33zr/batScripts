import wave
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft

def get_frequency(filename, freq_min, freq_max):
    with wave.open(filename, "rb") as wave_file:
        # Get the sample rate and number of frames
        sample_rate = wave_file.getframerate()
        num_frames = wave_file.getnframes()
        
        # Read the audio data from the wave file
        audio_data = wave_file.readframes(num_frames)
        audio_data = np.frombuffer(audio_data, dtype=np.int16)
        
        # Compute the FFT of the audio data
        fft_data = fft(audio_data)
        fft_data = np.abs(fft_data)
        fft_freqs = np.fft.fftfreq(len(fft_data), 1/sample_rate)
        
        # Get the indices of the desired frequency range
        start_idx = np.searchsorted(fft_freqs, freq_min)
        end_idx = np.searchsorted(fft_freqs, freq_max)
        
        # Get the frequency with the highest amplitude
        max_idx = np.argmax(fft_data[start_idx:end_idx])
        max_frequency = fft_freqs[start_idx + max_idx]
        
        return max_frequency, fft_data, fft_freqs

def plot_spectrogram(filename):
    with wave.open(filename, "rb") as wave_file:
        # Get the sample rate and number of frames
        sample_rate = wave_file.getframerate()
        num_frames = wave_file.getnframes()
        
        # Read the audio data from the wave file
        audio_data = wave_file.readframes(num_frames)
        audio_data = np.frombuffer(audio_data, dtype=np.int16)
        
        # Plot the spectrogram
        plt.specgram(audio_data, NFFT=1024, Fs=sample_rate)
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        plt.show()

def plot_amplitude_vs_frequency(fft_data, fft_freqs):
    plt.plot(fft_freqs, fft_data)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.show()

filename = "sample.wav"
freq_min = 0
freq_max = 1000

max_frequency, fft_data, fft_freqs = get_frequency(filename, freq_min, freq_max)
print("The frequency with the highest amplitude in the range [{}, {}] is {} Hz".format(freq_min, freq_max, max_frequency))
