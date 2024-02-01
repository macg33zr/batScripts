#
# Some utilities
#
import numpy as np
import os
import fnmatch
from tinytag import TinyTag
import re
import shutil
import logging
import time
import matplotlib.pyplot as plt
import scipy.signal as sp
from scipy.fft import *
from scipy import signal
import soundfile as sf
from ipyfilechooser import FileChooser

# Initialise logging to make a log file at a give path location
def initialise_logging(path):
    # Set up logging to a file in the path
    log_file_full_path = os.path.join(path, time.strftime("%Y%m%d-%H%M%S")+'_log.txt')
    print('Logging to:'+log_file_full_path)
    logging.basicConfig(filename=log_file_full_path, encoding='utf-8', level=logging.INFO,format='%(asctime)s %(message)s')
    logging.info('Processing directory:'+path)

# Log and print
def log_and_print(item):
    print(item)
    logging.info(item)

# Get a directory or file path using a browsing UI
def browse_for_path(path, only_directories = True):
    # Show a dialog to browse for the path
    fdialog = FileChooser(
        path,
        title='<b>Browse to Recordings to Process</b>',
        show_hidden=False,
        select_default=True,
        use_dir_icons=True,
        show_only_dirs=only_directories
    )
    display(fdialog)

    # Return the file dialog. Browsed item is fdialog.selected
    return fdialog

# Check if file is a WAV file and exists
def check_wav_file(filename):

    if not os.path.isfile(filename):
        if os.path.isdir(filename):
            print(f"Error: '{filename}' is a directory, not a file.")
            return False
        else:
            print(f"Error: '{filename}' does not exist.")
            return False

    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            header = f.read(4)
            if header == b'RIFF':
                print(f"{filename} exists and is a WAV file.")
                return True
            else:
                print(f"{filename} exists, but it is not a WAV file.")
                return False
    else:
        print(f"{filename} does not exist.")
        return False

# Load a .wav file. 
# These are 24 bit files. The PySoundFile library is able to read 24 bit files.
# https://pysoundfile.readthedocs.io/en/0.9.0/
def get_wav_info(wav_file):
    data, rate = sf.read(wav_file)
    return data, rate

# Find the list of files in a given path, sort by name
def find_and_sort_files_on_path(path):
    files = fnmatch.filter(os.listdir(path), "*.WAV")
    files.sort()
    print(files)
    logging.info('Found files: '+str(files))
    return files

#
# Get the comment tag and recorded time, voltage etc for a WAV file
#
def get_comment_and_data(wav_file_full_path):
    comment = ''
    recorded_time = ''
    temperature = ''
    battery = ''
    tag = TinyTag.get(wav_file_full_path)

    # Recorded at 01:09:00 07/09/2021 (UTC) by AudioMoth 247AA506603EC3A8 at medium-high gain while battery was 4.0V and temperature was 22.6C
    if not tag.comment is None:
        comment = tag.comment
        m = re.search('Recorded at (.+?) by AudioMoth (.+?) at (.+?) while battery was (.+?)V and temperature was (.+?)C', comment)
        if m:
            recorded_time = m.group(1)
            battery = m.group(4)
            temperature = m.group(5)

    return tag.comment, recorded_time, battery, temperature

# Initialise all the stats we used for processing
def initialise_processing_stats(files):
    # Stats item
    stats ={}

    # Stats for total files and unclassified
    stats['Total Files'] = len(files)
    stats['Unclassified'] = 0
    stats['MaxUnclassified'] = 0
    stats['MinUnclassified'] = 0

    # AudioMoth voltage and temperature stats from the files
    stats['MaxVolts'] = 0.0
    stats['MinVolts'] = 0.0
    stats['MaxTemp'] = 0.0
    stats['MinTemp'] = 0.0
    return stats

#
# Plot spectrogram and save to a file if supplied
#
def plot_spectrogram(data, rate, filename, recorded_time, do_plot = True, spectrogram_path=''):
    nfft = 2048
    fs = rate
    noverlap = 12
    nchannels = data.ndim
    
    plt.figure(figsize=(15.32, 7.49))
    if nchannels == 1:
        data = data
    elif nchannels == 2:
        data = data[:,0]
    
    # Suppress divide by zero warning on the data
    with np.errstate(divide='ignore'):
        plt.specgram(data, NFFT=nfft, Fs=fs, Fc=0, noverlap=noverlap, cmap=plt.cm.bone, sides='default', mode='default', scale='dB')    
    plt.title("Spectrogram - " + filename + " " + recorded_time)
    plt.xlabel('Time (s)')
    plt.ylabel('Freq (hz)')
    if(spectrogram_path != ''):
        plt.savefig(spectrogram_path, dpi=300, bbox_inches='tight')
    if(do_plot):
        plt.show() 

#
# Get max frequency in a given range
#
def get_max_frequency_in_range(signal, sample_rate, low_freq, high_freq):
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

#
# Getfreq max values using FFT
#
# 22Khz - 30 Khz
# 40Khz - 60 Khz
#
def plot_get_freqs(data, rate, filename, recorded_time, doPlot, startFreq, endFreq, freq_chart_path=''):
    N = len(data)
    yf = np.fft.rfft(data)
    xf = np.fft.rfftfreq(N, d=1./rate)
    
    # Filter out 32Hhz noise signal spike which I seem to get in the data
    f32khz = np.where(np.logical_and(xf>=32500.0, xf<=32900.0))
    if np.size(f32khz) > 0:
        yf[f32khz] = 0.0 
            
    # Filter out low frequency spike
    fLow = np.where(np.logical_and(xf>=0.0, xf<=1000.0))
    if np.size(fLow) > 0:
        yf[fLow] = 0.0    
        
    # Get max in the range (e.g. bat detection)
    finterest_max_amplitude = 0
    finterest = np.where(np.logical_and(xf>=startFreq, xf<=endFreq))
    if np.size(finterest) > 0:
        finterest_max_amplitude = np.max(np.abs(yf[finterest]))
  
    # Now get the max frequency on the FFT values bucket for the whole file
    idx = np.argmax(np.abs(yf))
    max_freq = xf[idx]  
    max_freq_val = np.abs(yf[idx])

    # Get the frequency with the max amplitude in the range. Can this be optimised with FFT above??
    finterest_max_frequency = get_max_frequency_in_range(data,rate, startFreq, endFreq)
    
    # Do out plot of frequencies with the 32Khz spike attenuated..
    plt.plot(xf, np.abs(yf))
    plt.title("Frequency chart - " + filename + " " + recorded_time)
    plt.xlabel('Freq (hz)')
    plt.ylabel('Value')
    if(freq_chart_path != ''):
        plt.savefig(freq_chart_path, dpi=300, bbox_inches='tight')

    if doPlot:
        plt.show()
    
    # Return the frequency and max val
    return max_freq, max_freq_val, finterest_max_amplitude, finterest_max_frequency




