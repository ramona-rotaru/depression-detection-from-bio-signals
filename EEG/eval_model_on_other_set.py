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
from EEGModels import *
from sklearn.model_selection import cross_val_score
from tensorflow.keras.utils import to_categorical
import sklearn
from tensorflow.python.keras.callbacks import ModelCheckpoint
import tensorflow as tf
import datetime

data_path = './eeg_control.npy'
model_path = '../code&data -- mumtaz_data/models_final/Mumtaz_0.6_BUN_EO_150_epochs_pat_40_split_80-20_val_6_bs128.h5'



eeg_data = np.load(data_path, allow_pickle=True)
print(eeg_data.shape)


one_subject = eeg_data[8]

sfreq = 500

ch_names = [f'EEG {i}' for i in range(one_subject.shape[0])]
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')


raw = mne.io.RawArray(one_subject, info)

raw = raw.resample(256, npad='auto')
print(raw.info)

raw.filter(l_freq=2, h_freq=40)
data = raw.get_data()
# raw.plot_psd()
print(data.shape)

epoch_len = 256
no_epochs = data.shape[1]//epoch_len

# Trim the data to fit into an exact number of epochs
data = data[:, :no_epochs * epoch_len]

print(no_epochs)
# epoched_data = np.empty(data.shape[0], data.shape[1]//epoch_len, epoch_len)

epoched_data = np.split(data, no_epochs, axis=1)
print(epoched_data[0].shape)
data_array = np.array(epoched_data)
print(data_array.shape)


#select some interemidiate epochs to evaluate model
# x_test = data_array[150:200]
x_test = data_array.swapaxes(1, 2)
y_test =  [0] * x_test.shape[0]
y_test = to_categorical(y_test, num_classes=2)



model = tf.keras.models.load_model(model_path)
print(model.summary())
print(model.evaluate(x_test, y_test), 'prediction')