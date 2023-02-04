import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

def get_frequency(signal, sample_rate, low_freq, high_freq):
    # Get the Fourier Transform of the signal
    fourier_transform = np.fft.fft(signal)
    frequencies = np.fft.fftfreq(signal.size, 1/sample_rate)

    # Get the amplitude of the Fourier Transform
    amplitude = np.abs(fourier_transform)
    
    # Get the frequencies in the specified range
    frequencies_in_range = frequencies[(frequencies >= low_freq) & (frequencies <= high_freq)]
    amplitude_in_range = amplitude[(frequencies >= low_freq) & (frequencies <= high_freq)]
    
    # Get the frequency with the highest amplitude
    max_amplitude_index = np.argmax(amplitude_in_range)
    frequency = frequencies_in_range[max_amplitude_index]
    
    return frequency

def plot_spectrogram(signal, sample_rate):
    plt.imshow(signal, origin='lower', aspect='auto')
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]')
    plt.title('Spectrogram of WAV File')
    plt.colorbar()
    plt.show()

def plot_amplitude_vs_frequency(frequencies, amplitude):
    plt.plot(frequencies, amplitude)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude')
    plt.title('Amplitude vs Frequency of WAV File')
    plt.show()

def graph_spectrogram(data, rate):
    nfft = 2048
    fs = rate
    noverlap = 12
    nchannels = data.ndim
    
    plt.figure(figsize=(15.32, 7.49))
    if nchannels == 1:
        data = data
    elif nchannels == 2:
        data = data[:,0]
    
    pxx, freqs, bins, im = plt.specgram(data, NFFT=nfft, Fs=fs, Fc=0, noverlap=noverlap, cmap=plt.cm.bone, sides='default', mode='default', scale='dB')
    
    

# Load the WAV file
sample_rate, signal = wavfile.read('pipistrelle_bat_recording.wav')

# Get the frequency with the highest amplitude in the specified range
frequency = get_frequency(signal, sample_rate, 0, 1000)
print(f'The frequency with the highest amplitude in the range [0, 1000] is {frequency} Hz')

# Get the spectrogram of the audio data
spectrogram = np.abs(np.fft.fft(signal))
plot_spectrogram(spectrogram, sample_rate)

# Get the amplitude versus frequency of the audio data
frequencies = np.fft.fftfreq(signal.size, 1/sample_rate)
amplitude = np.abs(np.fft.fft(signal))
plot_amplitude_vs_frequency(frequencies, amplitude)
