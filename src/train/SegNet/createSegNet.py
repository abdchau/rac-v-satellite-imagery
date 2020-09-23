'''
Created on Aug 29, 2018

@author: daniel
'''
import tensorflow.keras.backend as K
from tensorflow.python.keras.utils.data_utils import Sequence
import numpy as np
import random
import os
import pandas as pd

smooth = 1e-12
num_channels = 16
num_mask_channels = 3
img_rows = 128
img_cols = 128
cache_path = '../../cache'


from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.layers import Conv2D, Input, BatchNormalization, MaxPool2D, Activation, UpSampling2D, Softmax, Cropping2D
from tensorflow.keras.models import Sequential

def segnet_basic(input_shape=(128,128,16), n_labels=3, orig_depth=32):
    model = Sequential()
    model.add(Input(shape=input_shape))

    # contraction 1
    model.add(Conv2D(orig_depth, kernel_size=3, strides=1, kernel_initializer='he_normal', padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(orig_depth, kernel_size=3, strides=1, kernel_initializer='he_normal', padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=2))

    # contraction 2
    model.add(Conv2D(orig_depth*2, kernel_size=3, strides=1, kernel_initializer='he_normal', padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(orig_depth*2, kernel_size=3, strides=1, kernel_initializer='he_normal', padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=2))

    # contraction 3
    model.add(Conv2D(orig_depth*4, kernel_size=3, strides=1, kernel_initializer='he_normal', padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(orig_depth*4, kernel_size=3, strides=1, kernel_initializer='he_normal', padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(orig_depth*4, kernel_size=3, strides=1, kernel_initializer='he_normal', padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=2))

    # contraction 4
    model.add(Conv2D(orig_depth*8, kernel_size=3, strides=1, kernel_initializer='he_normal', padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(orig_depth*8, kernel_size=3, strides=1, kernel_initializer='he_normal', padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(orig_depth*8, kernel_size=3, strides=1, kernel_initializer='he_normal', padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=2))


    # expansion 1
    model.add(UpSampling2D(size=2))
    model.add(Conv2D(orig_depth*8, kernel_size=3, strides=1, kernel_initializer='he_normal', padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(orig_depth*8, kernel_size=3, strides=1, kernel_initializer='he_normal', padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(orig_depth*8, kernel_size=3, strides=1, kernel_initializer='he_normal', padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # expansion 2
    model.add(UpSampling2D(size=2))
    model.add(Conv2D(orig_depth*4, kernel_size=3, strides=1, kernel_initializer='he_normal', padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(orig_depth*4, kernel_size=3, strides=1, kernel_initializer='he_normal', padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(orig_depth*4, kernel_size=3, strides=1, kernel_initializer='he_normal', padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # expansion 3
    model.add(UpSampling2D(size=2))
    model.add(Conv2D(orig_depth*2, kernel_size=3, strides=1, kernel_initializer='he_normal', padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(orig_depth*2, kernel_size=3, strides=1, kernel_initializer='he_normal', padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    # expansion 4
    model.add(UpSampling2D(size=2))
    model.add(Conv2D(orig_depth, kernel_size=3, strides=1, kernel_initializer='he_normal', padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(orig_depth, kernel_size=3, strides=1, kernel_initializer='he_normal', padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(n_labels, kernel_size=1, padding='same'))# , activation='sigmoid'))
    model.add(Cropping2D(cropping=((16, 16), (16, 16))))
    model.add(Softmax())

    model.compile(optimizer=Nadam(lr=1e-3), loss='categorical_crossentropy', metrics=['acc'])
    return model

def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x

def save_model(model, cross):
    json_string = model.to_json()
    if not os.path.isdir(cache_path):
        os.mkdir(cache_path)
    json_name = 'architecture_' + cross + '.json'
    weight_name = 'model_weights_' + cross + '.h5'
    open(os.path.join(cache_path, json_name), 'w').write(json_string)
    model.save_weights(os.path.join(cache_path, weight_name), overwrite=True)


def save_history(history, suffix):
    filename = './history/history_' + suffix + '.csv'
    with open(filename, 'w') as f:
        pd.DataFrame(history.history).to_csv(f, index=False)

def append_history(history, suffix):
    filename = './history/history_' + suffix + '.csv'
    with open(filename, 'a') as f:
        pd.DataFrame(history.history).to_csv(f, index=False, header=False)

def read_model(cross='', load=True):
    json_name = 'architecture_' + cross + '.json'
    weight_name = 'model_weights_' + cross + '.h5'
    model = keras.models.model_from_json(open(os.path.join(cache_path, json_name)).read())
    if load:
        model.load_weights(os.path.join(cache_path, weight_name))
    return model

# creates a single batch; designed to be called by the generator
def form_batch(X, y, batch_size, min_true_label):
    X_batch = np.zeros((batch_size, num_channels, img_rows, img_cols))
    y_batch = np.zeros((batch_size, num_mask_channels, img_rows, img_cols))
    X_height = X.shape[2]
    X_width = X.shape[3]

    for i in range(batch_size):
        while True:
            random_width = random.randint(0, X_width - img_cols - 1)
            random_height = random.randint(0, X_height - img_rows - 1)

            random_image = random.randint(0, X.shape[0] - 1)
            y_batch[i] = y[random_image, :, random_height: random_height + img_rows, random_width: random_width + img_cols]
            X_batch[i] = np.array(X[random_image, :, random_height: random_height + img_rows, random_width: random_width + img_cols])

            # forces a minimum amount of true values in the random patches
            if min_true_label is not None:
                sum_ = np.sum(y_batch[:,2,:,:])
                # print(sum_,min_true_label*i, 'road')
                if sum_ >= min_true_label*i:
                    break

            else:
                break

    return X_batch, y_batch


# batch generator to use with model.fit_generator()
class MyGenerator(Sequence):

    def __init__(self, X, y, batch_size, horizontal_flip=False, vertical_flip=False, swap_axis=False, min_true_label=None):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.swap_axis = swap_axis
        self.min_true_label = min_true_label

    def __len__(self):
        return int(np.ceil(self.X.shape[0] / float(self.batch_size)))

    def __getitem__(self, idx):
        X_batch, y_batch = form_batch(self.X, self.y, self.batch_size, self.min_true_label)

        # apply random transformations to each patch within the batch
        for i in range(X_batch.shape[0]):
            xb = X_batch[i]
            yb = y_batch[i]

            if self.horizontal_flip:
                if np.random.random() < 0.5:
                    xb = flip_axis(xb, 1)
                    yb = flip_axis(yb, 1)

            if self.vertical_flip:
                if np.random.random() < 0.5:
                    xb = flip_axis(xb, 2)
                    yb = flip_axis(yb, 2)

            if self.swap_axis:
                if np.random.random() < 0.5:
                    xb = xb.swapaxes(1, 2)
                    yb = yb.swapaxes(1, 2)

            X_batch[i] = xb
            y_batch[i] = yb

        #print(X_batch.shape, y_batch.shape)

        X_batch = X_batch.swapaxes(1,3)
        y_batch = y_batch.swapaxes(1,3)

        #print(X_batch.shape, y_batch.shape)

        return X_batch, y_batch[:, 16:16 + img_cols - 32, 16:16 + img_rows - 32, :]