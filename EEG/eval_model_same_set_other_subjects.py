from utils import *
from EEGModels import EEGNet, Mumtaz_model_attention,Mumtaz_model_attention_2, DeepConvNet, DepressioNet, ceva, EEGDepressionNet, ModerateEEGDepressionNet, CustomEEGModel, Mumtaz_model_orig
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
import seaborn as sns

path_to_pre_processed_data = './data/'


all_healthy = [f for f in os.listdir(data_path) if f.startswith('H') and f.endswith('EO.edf')]
all_mdd = [f for f in os.listdir(data_path) if f.startswith('MDD') and f.endswith('EO.edf')]
s_f = [filter_data(i) for i in all_healthy]
b_f = [filter_data(i) for i in all_mdd]

# print(len(s_f)) 
# print(len(b_f))

sf_train = s_f[:int(len(s_f)*0.8)]
sf_test = s_f[int(len(s_f)*0.8):]


bf_train = b_f[:int(len(b_f)*0.8)]
bf_test = b_f[int(len(b_f)*0.8):]


concat_matrix_s_train = np.concatenate(sf_train, axis=0)
concat_matrix_b_train = np.concatenate(bf_train, axis=0)
concat_matrix_train = np.concatenate((concat_matrix_s_train, concat_matrix_b_train), axis=0)
print(concat_matrix_train.shape, 'concat_matrix_train.shape')

concat_matrix_s_test = np.concatenate(sf_test, axis=0)
concat_matrix_b_test = np.concatenate(bf_test, axis=0)
concat_matrix_test = np.concatenate((concat_matrix_s_test, concat_matrix_b_test), axis=0)
print(concat_matrix_test.shape, 'concat_matrix_test.shape')


#calculate labels
no_healthy_train = 0
for subject in sf_train:
    print(subject.shape[0]) 
    no_healthy_train = no_healthy_train + subject.shape[0]
    
    
no_healthy_test = 0
for subject in sf_test:
    print(subject.shape[0]) 
    no_healthy_test = no_healthy_test + subject.shape[0]
    
no_sick_train = concat_matrix_train.shape[0] - no_healthy_train
no_sick_test = concat_matrix_test.shape[0] - no_healthy_test


labels_healthy_train = [0] * no_healthy_train
labels_healthy_test = [0] * no_healthy_test

labels_sick_train = [1] * no_sick_train
labels_sick_test = [1] * no_sick_test

labels_train = np.concatenate((labels_healthy_train, labels_sick_train), axis=0)
labels_test = np.concatenate((labels_healthy_test, labels_sick_test), axis=0)


#shuffle data
x_train, y_train = shuffle(concat_matrix_train, labels_train, random_state=33)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[2], x_train.shape[1])
x_test, y_test = shuffle(concat_matrix_test, labels_test, random_state=33)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[2], x_test.shape[1])


y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


print(x_train.shape, 'x_train.shape')
nb_classes = 2


directory = './models_final/'
model_name = 'Mumtaz_0.6_BUN_EC_150_epochs_pat_40_split_80-20_val_6_bs128'

load_model_path = directory + model_name + '.h5'


# load_model_path = 'additional_models+phoes\Mumtaz_articol_original_TASK100_epochs_pat_20_split_80-20_val_6.h5'
model = tf.keras.models.load_model(load_model_path)
model.summary()
print(model.evaluate(x_test, y_test), 'prediction')


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
cm = confusion_matrix(y_test, y_pred)
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
sns.heatmap(cm, annot=True, cmap="BuPu", fmt='d', 
            xticklabels=['Control', 'TDM'], yticklabels=['Control', 'TDM'], annot_kws={"size": 17})
plt.xlabel('Predicted', fontsize=17)
plt.ylabel('Truth', fontsize=17)
plt.title('Confusion Matrix', fontsize=17)
plt.savefig('./models_final/confusion_matrix_nenorm' + model_name + '.png', dpi=300, bbox_inches='tight')
plt.show()

