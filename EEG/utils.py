import numpy as np
import mne
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.stats import median_abs_deviation, skew, kurtosis
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import itertools
import pandas as pd

from sklearn.model_selection import cross_val_score
from tensorflow.keras.utils import to_categorical
import sklearn
from tensorflow.python.keras.callbacks import ModelCheckpoint
import tensorflow as tf
import datetime
from mne_icalabel import label_components
from tqdm import tqdm
from sklearn.utils import shuffle
# from mne_icalabel.gui import label_ica_components


# data_path = './eeg_preprocessed_new/'
data_path = 'data/'


def preprocessing(data_path_1instance, l_freq=0.2, h_freq=40, no_components=19, random_state=97, duration=1, overlap=0):
    """
    Preprocess the data.
    
    Args:
        data_path (str): Path to the data.
        l_freq (float): Low cut-off frequency.
        h_freq (float): High cut-off frequency.
        no_components (int): Number of components for ICA.
        random_state (int): Random state for ICA.
        duration (float): Duration of the epochs.
        overlap (float): Overlap of the epochs.
        
    Returns:
        array: Preprocessed data.
    """
    raw = mne.io.read_raw_edf(data_path_1instance, preload=True)

    if 'EEG 24A-24R' in raw.ch_names:
        raw.drop_channels(['EEG 24A-24R'])
        
    if 'EEG 23A-23R' in raw.ch_names:
        raw.drop_channels(['EEG 23A-23R'])
        
    if 'EEG A2-A1' in raw.ch_names:
        raw.drop_channels(['EEG A2-A1'])
      
    # raw.drop_channels(['EEG A2-A1', 'EEG 23A-23R', 'EEG 24A-24R'])
    raw.rename_channels(lambda x: x.strip('EEG '))
    raw.rename_channels(lambda x: x.strip('-LE'))
    raw.set_montage('standard_1020')
    
    filter_params = mne.filter.create_filter(
        raw.get_data(), raw.info["sfreq"], l_freq, h_freq
    )
    filtered = raw.filter(l_freq, h_freq)
    raw.notch_filter(50, verbose=False)
    filtered = filtered.set_eeg_reference("average")
    
    ica = mne.preprocessing.ICA(
        n_components=no_components,
        max_iter="auto",
        method="infomax",
        random_state=random_state,
        fit_params=dict(extended=True),
    )
    print('Fitting ICA... for: ' + data_path_1instance)
    ica.fit(filtered)
    ic_labels = label_components(filtered, ica, method="iclabel")
    labels = ic_labels["labels"]
    exclude_idx = [idx for idx, label in enumerate(labels) if label not in ["brain",  "other"]]
    reconst_filt = filtered.copy()
    ica.apply(reconst_filt, exclude=exclude_idx)
    epochs = mne.make_fixed_length_epochs(reconst_filt, duration=duration, overlap=overlap)
    # epochs.apply_baseline((0, 1))
    
    array = epochs.get_data()
    print('Done preprocessing for: ' + data_path_1instance)
    return array
    


def plot_fft(signal, sample_rate):
    """
    Plot the FFT response of a signal.
    
    Args:
        signal (array-like): Input signal.
        sample_rate (float): Sampling rate of the signal.
    """
    # Compute the FFT
    fft_values = np.fft.fft(signal)
    fft_freq = np.fft.fftfreq(len(signal), 1 / sample_rate)
    
    # Plot the FFT response
    plt.figure(figsize=(8, 5))
    plt.plot(fft_freq, np.abs(fft_values))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title('FFT Response')
    plt.grid(True)
    plt.show()


def plot_psd(signal, sample_rate):
    """
    Plot the Power Spectral Density (PSD) of a signal.
    
    Args:
        signal (array-like): Input signal.
        sample_rate (float): Sampling rate of the signal.
    """
    plt.figure(figsize=(8, 5))
    plt.psd(signal, Fs=sample_rate)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density (dB/Hz)')
    plt.title('Power Spectral Density (PSD)')
    plt.grid(True)
    plt.show()



def read_eo_ec_data(filename):
    raw = mne.io.read_raw_edf(data_path + filename, preload=True)
    array = raw.get_data()
    return array


def read_data(filename):
    raw = mne.io.read_raw_edf(data_path + filename, preload=True)
    # raw.drop_channels(['EEG A2-A1', 'EEG 23A-23R', 'EEG 24A-24R'])
    epochs = mne.make_fixed_length_epochs(raw, duration=1, ocverlap=0)
    array = epochs.get_data()
    return array


def filter_data(filename):
   data = mne.io.read_raw_edf(data_path + filename, preload=True)
   if data.ch_names[-1] == 'EEG 24A-24R':
        data.drop_channels(['EEG A2-A1', 'EEG 23A-23R', 'EEG 24A-24R'])
   else:
        data.drop_channels(['EEG A2-A1'])

  #  filter_params = mne.filter.create_filter(data.get_data(), data.info["sfreq"],l_freq=2, h_freq=40)
  #  mne.viz.plot_filter(filter_params, data.info["sfreq"],)

   data.filter(l_freq=2, h_freq=40)
   epochs = mne.make_fixed_length_epochs(data, duration=1, overlap=0)
   array = epochs.get_data()
   print(array.shape, 'array.shape')
   array = array.swapaxes(1, 2)#reshape array to fit mumtaz model
   return array



def get_tensorboard_callback(model_name):
#     log_dir = "log/log" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    #SAVE LOG AS MODEL NAME PLUS DATE AND TIME
    log_dir = "log/" + model_name + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    return tensorboard_callback

