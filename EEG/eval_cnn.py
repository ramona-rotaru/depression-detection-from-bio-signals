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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
import seaborn as sns
from utils import *



tf.config.experimental.enable_tensor_float_32_execution(False)
for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

data_path = 'data/'
# all_healthy = [f for f in os.listdir(data_path) if f.startswith('H') and f.endswith('TASK.edf')]
# all_mdd = [f for f in os.listdir(data_path) if f.startswith('MDD') and f.endswith('TASK.edf')]# 

all_healthy = [f for f in os.listdir(data_path) if f.startswith('H') and f.endswith('TASK.edf')]
all_mdd = [f for f in os.listdir(data_path) if f.startswith('MDD') and f.endswith('TASK.edf')]

# print(len(all_healthy))
# print(len(all_mdd))

# Load data
s_f = [filter_data(i) for i in all_healthy]
b_f = [filter_data(i) for i in all_mdd]

# print(len(s_f)) 
# print(len(b_f))
# print(s_f[0].shape)

concat_matrix_s = np.concatenate(s_f, axis=0)
concat_matrix_b = np.concatenate(b_f, axis=0)
concat_matrix = np.concatenate((concat_matrix_s, concat_matrix_b), axis=0)
print(concat_matrix.shape, 'concat_matrix.shape')


no_healthy = 0
for subject in s_f:
  print(subject.shape[0]) 
  no_healthy = no_healthy + subject.shape[0]

# print(no_healthy)

labels_healthy = [0] * no_healthy

labels_sick = [1] * (concat_matrix.shape[0] - no_healthy)
print(len(labels_sick), 'labels_sick')
labels = np.concatenate((labels_healthy, labels_sick), axis=0)

print(len(labels_healthy), 'labels_healthy')
print(len(labels_sick), 'labels_sick')

print(labels.shape, 'labels.shape')
print(concat_matrix.shape, 'concat_matrix.shape')

# concat_matrix = concat_matrix.reshape(concat_matrix.shape[0], concat_matrix.shape[2], concat_matrix.shape[1])
# print(concat_matrix.shape, 'concat_matrix.shape')

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(concat_matrix, labels, test_size=0.2,
                                                                                    train_size=0.8, random_state=33,
                                                                                    shuffle=True, stratify=labels)


y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


print(x_train.shape, 'x_train.shape')
nb_classes = 2


# load_model_path = 'additional_models+phoes/Mumtaz_model_orig_TASK_65_epochs_80-20_sigmoid_96acc_bs8.h5'
# load_model_path ='additional_models+phoes\Mumtaz_model_with_attention_TASK_65_epochs_80-20_sigmoid_tanh_bs8.h5'
load_model_path = 'additional_models+phoes\Mumtaz_model_with_attention_TASK_65_epochs_80-20_sigmoid_94acc_bs8.h5'
model = tf.keras.models.load_model(load_model_path)
# print(model.evaluate(x_test, to_categorical(y_test)), 'prediction')
y_pred = model.predict(x_test)

y_pred=np.argmax(y_pred, axis=1)
y_test=np.argmax(y_test, axis=1)

#calculate accuracy score
print('Accuracy: ', accuracy_score(y_test, y_pred)*100, '%')
# Calculate precision
precision = precision_score(y_test, y_pred)
print("Precision:", precision)

# Calculate recall
recall = recall_score(y_test, y_pred)
print("Recall:", recall)

# Calculate F1-score
f1 = f1_score(y_test, y_pred)
print("F1-score:", f1)
#do confusion matrix
cm = confusion_matrix(y_test, y_pred, )
print(cm)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

# Calculate specificity (true negative rate)
specificity = tn / (tn + fp)
print("Specificity:", specificity)

# Calculate sensitivity (true positive rate)
sensitivity = tp / (tp + fn)
print("Sensitivity:", sensitivity)


#plot confusion matrix
plt.figure()
sns.heatmap(cm, annot=True, cmap="BuPu", fmt='d', xticklabels=['Control', 'TDM'], yticklabels=['Control', 'TDM'], annot_kws={"size": 16})
plt.xlabel('Predicted', fontsize=16)
plt.ylabel('Truth', fontsize=16)
plt.title('Confusion Matrix', fontsize=16)
plt.show()


