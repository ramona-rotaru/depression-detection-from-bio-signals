
from utils import *
from EEGModels import *

data_path = 'data/'
# all_healthy = [f for f in os.listdir(data_path) if f.startswith('H') and f.endswith('TASK.edf')]
# all_mdd = [f for f in os.listdir(data_path) if f.startswith('MDD') and f.endswith('TASK.edf')]# 


all_healthy = [f for f in os.listdir(data_path) if f.startswith('H') and f.endswith('EO.edf')]
all_mdd = [f for f in os.listdir(data_path) if f.startswith('MDD') and f.endswith('EO.edf')]

# Load and filter data

s_f = [filter_data(i) for i in all_healthy]
b_f = [filter_data(i) for i in all_mdd]

# print(len(s_f)) 
# print(len(b_f))


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

concat_matrix = concat_matrix.reshape(concat_matrix.shape[0], concat_matrix.shape[2], concat_matrix.shape[1])
# print(concat_matrix.shape, 'concat_matrix.shape')

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(concat_matrix, labels, test_size=0.2,
                                                                                    train_size=0.8, random_state=33,
                                                                                    shuffle=True, stratify=labels)


y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


print(x_train.shape, 'x_train.shape')
nb_classes = 2


model = Mumtaz_model_attention()

optimizer_ = tf.keras.optimizers.Adam(0.0003)
model.compile(loss="binary_crossentropy", optimizer=optimizer_, metrics=['accuracy'])
print(model.summary())

save_model_path = 'Mumtaz_model_with_attention_EC_65_epochs_80-20_sigmoid_tanh_bs8.h5'

model.fit(x_train, y_train, batch_size=128, epochs=65, verbose=2, callbacks=[get_tensorboard_callback(str(save_model_path)), tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=30, verbose=0, mode="auto", baseline=None, restore_best_weights=False), ModelCheckpoint(save_model_path, save_best_only=True, verbose=1)], validation_data=(x_test, y_test))

# Train the model
model.fit(x_train, y_train)

load_model_path = save_model_path
model = tf.keras.models.load_model(load_model_path)
print(model.evaluate(x_test, y_test), 'prediction')





