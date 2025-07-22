from utils import *
from EEGModels import EEGNet, Mumtaz_model_attention,Mumtaz_model_attention_2, DeepConvNet, DepressioNet, ceva, EEGDepressionNet, ModerateEEGDepressionNet, CustomEEGModel, Mumtaz_model_orig


path_to_pre_processed_data = './data/'


all_healthy = [f for f in os.listdir(data_path) if f.startswith('H') and f.endswith('EC.edf')]
all_mdd = [f for f in os.listdir(data_path) if f.startswith('MDD') and f.endswith('EC.edf')]
s_f = [filter_data(i) for i in all_healthy]
b_f = [filter_data(i) for i in all_mdd]

# print(len(s_f)) 
# print(len(b_f))
# print(s_f[0].shape)
# exit() 

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

model = Mumtaz_model_attention()


model.compile(loss="binary_crossentropy", optimizer='adam', metrics=['accuracy'])
print(model.summary())

folder_path = './models_final/'
save_model_path = folder_path + 'Mumtaz_0.6_BUN_EC_150_epochs_pat_40_split_80-20_val_6_bs128.h5'


model.fit(x_train, y_train, batch_size=128, epochs=150, verbose=2, callbacks=[get_tensorboard_callback(str(save_model_path)), tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=40, verbose=0, mode="auto", baseline=None, restore_best_weights=False), ModelCheckpoint(save_model_path, save_best_only=True, verbose=1)], validation_data=(x_test, y_test))

# Train the model
model.fit(x_train, y_train)

load_model_path = save_model_path 
model = tf.keras.models.load_model(load_model_path)
print(model.evaluate(x_test, y_test), 'prediction')





