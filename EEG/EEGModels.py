
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Permute, Dropout, GlobalAveragePooling1D, LSTM, Reshape,  Bidirectional
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Conv1D, MaxPooling1D
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D, MultiHeadAttention
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import backend as K

from tensorflow.keras import layers


from tensorflow.keras.layers import Attention, Multiply
from tensorflow.keras.models import Model


def DepressioNet(classes=2, channels=19, samples=256*16 ,N1=8, d=2, kernelLength=32):
  input = Input(shape = (channels, samples))
  x = Permute((2,1))(input)
  x = Reshape((1, samples, channels))(x)
  x = Conv2D(filters=N1, kernel_size=(channels, 1), strides=1, padding='same')(x)
  x = DepthwiseConv2D(kernel_size=(1, kernelLength), strides=1, padding='valid', depth_multiplier=d)(x)
  x = Activation('relu')(x)
  x = BatchNormalization()(x)
  x = AveragePooling2D(pool_size=(1, samples-kernelLength+1))(x)
  x = Reshape((-1,))(x)
  out = Dense(classes, activation='softmax')(x)

  return Model(inputs=input, outputs=out)


def test(input_shape=(256, 19)):
    model = Sequential()

    # Convolutional layers
    model.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(Conv1D(64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Conv1D(128, kernel_size=3, activation='relu'))
    model.add(Conv1D(128, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # Recurrent layers
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(Bidirectional(LSTM(64)))

    # Fully connected layers
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))

    # Output layer
    model.add(Dense(1, activation='sigmoid'))  # for binary classification

    return model


def DeepConvNet(nb_classes, Chans = 64, Samples = 256,
                dropoutRate = 0.5):
    """ Keras implementation of the Deep Convolutional Network as described in
    Schirrmeister et. al. (2017), Human Brain Mapping.
    
    This implementation assumes the input is a 2-second EEG signal sampled at 
    128Hz, as opposed to signals sampled at 250Hz as described in the original
    paper. We also perform temporal convolutions of length (1, 5) as opposed
    to (1, 10) due to this sampling rate difference. 
    
    Note that we use the max_norm constraint on all convolutional layers, as 
    well as the classification layer. We also change the defaults for the
    BatchNormalization layer. We used this based on a personal communication 
    with the original authors.
    
                      ours        original paper
    pool_size        1, 2        1, 3
    strides          1, 2        1, 3
    conv filters     1, 5        1, 10
    
    Note that this implementation has not been verified by the original 
    authors. 
    
    """

    # start the model
    input_main   = Input((Chans, Samples, 1))
    block1       = Conv2D(25, (1, 5), 
                                 input_shape=(Chans, Samples, 1),
                                 kernel_constraint = max_norm(2., axis=(0,1,2)))(input_main)
    block1       = Conv2D(25, (Chans, 1),
                                 kernel_constraint = max_norm(2., axis=(0,1,2)))(block1)
    block1       = BatchNormalization(epsilon=1e-05, momentum=0.9)(block1)
    block1       = Activation('elu')(block1)
    block1       = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block1)
    block1       = Dropout(dropoutRate)(block1)
  
    block2       = Conv2D(50, (1, 5),
                                 kernel_constraint = max_norm(2., axis=(0,1,2)))(block1)
    block2       = BatchNormalization(epsilon=1e-05, momentum=0.9)(block2)
    block2       = Activation('elu')(block2)
    block2       = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block2)
    block2       = Dropout(dropoutRate)(block2)
    
    block3       = Conv2D(100, (1, 5),
                                 kernel_constraint = max_norm(2., axis=(0,1,2)))(block2)
    block3       = BatchNormalization(epsilon=1e-05, momentum=0.9)(block3)
    block3       = Activation('elu')(block3)
    block3       = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block3)
    block3       = Dropout(dropoutRate)(block3)
    
    block4       = Conv2D(200, (1, 5),
                                 kernel_constraint = max_norm(2., axis=(0,1,2)))(block3)
    block4       = BatchNormalization(epsilon=1e-05, momentum=0.9)(block4)
    block4       = Activation('elu')(block4)
    block4       = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block4)
    block4       = Dropout(dropoutRate)(block4)
    
    flatten      = Flatten()(block4)
    
    dense        = Dense(nb_classes, kernel_constraint = max_norm(0.5))(flatten)
    softmax      = Activation('softmax')(dense)
    
    return Model(inputs=input_main, outputs=softmax)



def EEGNet(nb_classes, Chans=64, Samples=128,
           dropoutRate=0.5, kernLength=64, F1=8,
           D=2, F2=16, norm_rate=0.25, dropoutType='Dropout'):


    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')

    input1 = Input(shape=(Chans, Samples, 1))

    block1 = Conv2D(F1, (1, kernLength), padding='same',
                    input_shape=(Chans, Samples, 1),
                    use_bias=False)(input1)
    block1 = BatchNormalization()(block1)

    block1 = DepthwiseConv2D((Chans, 1), use_bias=False,
                             depth_multiplier=D,
                             depthwise_constraint=max_norm(1.))(block1)
    block1 = BatchNormalization()(block1)
    block1 = Activation('elu')(block1)
    block1 = AveragePooling2D((1, 4))(block1)
    block1 = dropoutType(dropoutRate)(block1)

    block2 = SeparableConv2D(F2, (1, 16),
                             use_bias=False, padding='same')(block1)
    block2 = BatchNormalization()(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D((1, 8))(block2)
    block2 = dropoutType(dropoutRate)(block2)

    flatten = Flatten(name='flatten')(block2)

    dense = Dense(nb_classes, name='dense',
                  kernel_constraint=max_norm(norm_rate))(flatten)
    softmax = Activation('softmax', name='softmax')(dense)

    return Model(inputs=input1, outputs=softmax)



def CustomEEGModel(input_shape=(256, 19)):
    model = Sequential()

    # Convolutional Layers
    model.add(Conv1D(16, kernel_size=3, input_shape=input_shape, activation='tanh', padding='same'))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Conv1D(32, kernel_size=5, activation='tanh', padding='same'))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Conv1D(64, kernel_size=5, activation='tanh', padding='same'))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Conv1D(128, kernel_size=5, activation='tanh', padding='same'))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Conv1D(256, kernel_size=5, activation='tanh', padding='same'))
    model.add(MaxPooling1D(pool_size=2))

    # Flatten layer
    model.add(Flatten())

    # Fully connected layers
    model.add(Dense(512, activation='tanh'))
    model.add(Dropout(0.5))

    model.add(Dense(256, activation='tanh'))
    model.add(Dropout(0.5))

    model.add(Dense(128, activation='tanh'))

    # Output layer - for binary classification
    model.add(Dense(2, activation='sigmoid'))  

    return model

# Creating an instance of the model


def ModerateEEGDepressionNet(input_shape=(256, 19)):
    model = Sequential()

    # 0-1 Convolution
    model.add(Conv1D(16, kernel_size=5, input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # 1-2 Max-pooling
    model.add(MaxPooling1D(pool_size=2, strides=2))
    
    model.add(Dropout(0.5))


    # 2-3 Convolution
    model.add(Conv1D(32, kernel_size=5))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # 3-4 Max-pooling
    model.add(MaxPooling1D(pool_size=2, strides=2))
    
    model.add(Dropout(0.5))

    # 4-5 Convolution
    model.add(Conv1D(32, kernel_size=5))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(Dropout(0.5))

    # 5-6 Max-pooling
    model.add(MaxPooling1D(pool_size=2, strides=2))

    # Flatten before fully connected layers
    model.add(Flatten())

    # 6-7 Fully-connected
    model.add(Dense(82))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    # model.add(Dropout(0.3))

    # 7-8 Fully-connected
    model.add(Dense(64))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    # model.add(Dropout(0.5))

    # 8-9 Fully-connected (Output layer)
    model.add(Dense(2, activation='sigmoid'))  # For binary classification
    return model


def EEGDepressionNet(input_shape=(256, 19)):
    model = Sequential()

    # 0-1 Convolution
    model.add(Conv1D(5, kernel_size=5, activation='tanh', input_shape=input_shape))

    # 1-2 Max-pooling
    model.add(MaxPooling1D(pool_size=2, strides=2))

    # 2-3 Convolution
    model.add(Conv1D(5, kernel_size=5, activation='tanh'))

    # 3-4 Max-pooling
    model.add(MaxPooling1D(pool_size=2, strides=2))

    # 4-5 Convolution
    model.add(Conv1D(10, kernel_size=5, activation='tanh'))

    # 5-6 Max-pooling
    model.add(MaxPooling1D(pool_size=2, strides=2))

    # 6-7 Convolution
    model.add(Conv1D(10, kernel_size=5, activation='tanh'))

    # 7-8 Max-pooling
    model.add(MaxPooling1D(pool_size=2, strides=2))

    # 8-9 Convolution
    model.add(Conv1D(15, kernel_size=5, activation='tanh'))

    # 9-10 Max-pooling
    model.add(MaxPooling1D(pool_size=2, strides=2))

    # Flatten before fully connected layers
    model.add(Flatten())

    # 10-11 Fully-connected
    model.add(Dense(80, activation='sigmoid'))

    # 11-12 Fully-connected
    model.add(Dense(40, activation='sigmoid'))

    # 12-13 Fully-connected (Output layer)
    model.add(Dense(2, activation='sigmoid'))  # For binary classification

    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model


def Mumtaz_model_orig(input_shape=(256, 19)):

    # Input layer
    input1 = layers.Input(shape=input_shape)
    # First block
    block1 = layers.Conv1D(50, 10)(input1)
    # print(block1.shape)
    block1 = layers.Conv1D(50, 10)(block1)
    # print(block1.shape, 'before pooling')
    block1 = layers.MaxPooling1D(pool_size=3)(block1)
    # print(block1.shape, 'after pooling')
    # Second block
    block2 = layers.Conv1D(100, 10)(block1)
    block2 = layers.MaxPooling1D(3)(block2)

    # Third block
    block3 = layers.Conv1D(50, 21)(block2)

    # attention = layers.MultiHeadAttention(num_heads=2, key_dim=2)(block3, block3)

    # Global average pooling
    # gap = layers.GlobalAveragePooling1D()(attention)
    gap = layers.GlobalAveragePooling1D()(block3)

    # Dropout layer
    dropout = layers.Dropout(0.6)(gap)
    

    # Output layer
    output = layers.Dense(2, activation='sigmoid')(dropout)

    # Create the model
    model = tf.keras.Model(inputs=input1, outputs=output)

    return model


def Mumtaz_model_attention():
    input_shape = (256, 19)
    # Input layer
    input1 = layers.Input(shape=input_shape)
    # First block
    block1 = layers.Conv1D(50, 10)(input1)
    block1 = layers.Conv1D(50, 10)(block1)
    block1 = layers.MaxPooling1D(pool_size=3)(block1)

    
    # Second block
    block2 = layers.Conv1D(100, 10)(block1)
    block2 = layers.MaxPooling1D(3)(block2)

    # Third block
    block3 = layers.Conv1D(50, 21)(block2)
    # attention = layers.MultiHeadAttention(num_heads=2, key_dim=2)(block3, block3)

    # Global average pooling
    # gap = layers.GlobalAveragePooling1D()(attention)
    gap = layers.GlobalAveragePooling1D()(block3)

    # Dropout layer
    dropout = layers.Dropout(0.6)(gap)
    

    # Output layer
    output = layers.Dense(2, activation='sigmoid')(dropout)

    # Create the model
    model = tf.keras.Model(inputs=input1, outputs=output)
    return model



def Mumtaz_model_attention_2():
    input_shape = (256, 19)
    # Input layer
    input1 = layers.Input(shape=input_shape)
    # First block
    block1 = layers.Conv1D(50, 10, activation='tanh')(input1)
    block1 = layers.MultiHeadAttention(num_heads=2, key_dim=2)(block1, block1)
    # print(block1.shape)
    block1 = layers.Conv1D(50, 10, activation='tanh')(block1)
    # print(block1.shape, 'before pooling')
    block1 = layers.MaxPooling1D(pool_size=3)(block1)
    # print(block1.shape, 'after pooling')
    # Second block
    block2 = layers.Conv1D(100, 10, activation='tanh')(block1)
    block2 = layers.MaxPooling1D(3)(block2)
    block2 = layers.MultiHeadAttention(num_heads=2, key_dim=2)(block2, block2)
    # Third block
    block3 = layers.Conv1D(50, 21, activation='tanh')(block2)
    block3 = layers.MultiHeadAttention(num_heads=2, key_dim=2)(block3, block3)

    # Global average pooling
    # gap = layers.GlobalAveragePooling1D()(attention)
    gap = layers.GlobalAveragePooling1D()(block3)

    # Dropout layer
    dropout = layers.Dropout(0.5)(gap)


    # Output layer
    output = layers.Dense(2, activation='sigmoid')(dropout)

    # Create the model
    model = tf.keras.Model(inputs=input1, outputs=output)
    return model





def EEGNetWithAttentionAndLSTM(nb_classes, Chans=64, Samples=128,
                               dropoutRate=0.5, kernLength=64, F1=8,
                               D=2, norm_rate=0.25, dropoutType='Dropout'):

    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')

    input1 = Input(shape=(Chans, Samples, 1))

    block1 = Conv2D(F1, (1, kernLength), padding='same',
                    input_shape=(Chans, Samples, 1),
                    use_bias=False)(input1)
    block1 = BatchNormalization()(block1)

    block1 = DepthwiseConv2D((Chans, 1), use_bias=False,
                             depth_multiplier=D,
                             depthwise_constraint=max_norm(1.))(block1)
    block1 = BatchNormalization()(block1)
    block1 = Activation('elu')(block1)
    block1 = AveragePooling2D((1, 4))(block1)
    block1 = dropoutType(dropoutRate)(block1)

    # Attention mechanism
    attention = Attention()([block1, block1])
    block1 = Multiply()([block1, attention])

    lstm_units = 64  # Define the number of LSTM units

    # Reshape the tensor for LSTM


    # Add LSTM layer
    lstm = LSTM(units=lstm_units)(block1)

    dense = Dense(nb_classes, name='dense',
                  kernel_constraint=max_norm(norm_rate))(lstm)
    softmax = Activation('softmax', name='softmax')(dense)

    return Model


def EEGNetHybrid(nb_classes, Chans=19, Samples=256,
                               dropoutRate=0.5, kernLength=128, F1=8,
                               D=2, norm_rate=0.25, dropoutType='Dropout'):

    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')

    input1 = Input(shape=(Chans, Samples, 1))

    block1 = Conv2D(F1, (1, kernLength), padding='same',
                    input_shape=(Chans, Samples, 1),
                    use_bias=False)(input1)
    block1 = BatchNormalization()(block1)
    block1 = Activation('relu')(block1)
    block2 = DepthwiseConv2D((Chans, 1), use_bias=False,
                             depth_multiplier=D,
                             depthwise_constraint=max_norm(1))(block1)
    # block1 = BatchNormalization()(block1)
    block2 = Activation('relu')(block2)
    block2 = AveragePooling2D((1, 4))(block2)
    print(block2.shape)

    block3 = MultiHeadAttention(num_heads=2, key_dim=2, attention_axes=(1, 2))(block2, block2)
    print(block3.shape)


    dense = Dense(nb_classes, name='dense',
                  kernel_constraint=max_norm(norm_rate))(block3)
    
    softmax = Activation('sigmoid', name='sigmoid')(dense)

    return Model(inputs=input1, outputs=softmax)