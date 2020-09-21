'''
Created on Aug 29, 2018

@author: daniel
'''
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Activation, Conv2D, BatchNormalization, Cropping2D
from CustomLayers.MaxPoolingWithArgmax2D import MaxPoolingWithArgmax2D
from CustomLayers.MaxUnpooling2D import MaxUnpooling2D
import tensorflow.keras.backend as K
from tensorflow.python.keras.utils.data_utils import Sequence
from tensorflow.keras.optimizers import Nadam
import numpy as np
import random
import os
import pandas as pd

smooth = 1e-12
num_channels = 16
num_mask_channels = 2
img_rows = 96
img_cols = 96
cache_path = '../../cache'

def jaccard_coef(y_true, y_pred):
    intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])

    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return K.mean(jac)


def jaccard_coef_int(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))

    intersection = K.sum(y_true * y_pred_pos, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred_pos, axis=[0, -1, -2])

    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return K.mean(jac)


def jaccard_coef_loss(y_true, y_pred):
    return -K.log(jaccard_coef(y_true, y_pred)) + K.binary_crossentropy(y_pred, y_true)

def createSegNet(input_shape, 
               n_labels, 
               numFilters = 32,
               output_mode="softmax"):
        
    inputs = Input(shape=input_shape)

    conv_1 = Conv2D(numFilters, (3,3), padding="same", kernel_initializer = 'he_normal')(inputs)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Activation("relu")(conv_1)

    pool_1, mask_1 = MaxPoolingWithArgmax2D(pool_size=(2, 2))(conv_1)
    
    conv_2 = Conv2D(2*numFilters, (3,3), padding="same", kernel_initializer = 'he_normal')(pool_1)
    conv_2 = BatchNormalization()(conv_2)
    conv_2 = Activation("relu")(conv_2)

    pool_2, mask_2 = MaxPoolingWithArgmax2D(pool_size=(2, 2))(conv_2)
    
    conv_3 = Conv2D(2*numFilters, (3,3), padding="same", kernel_initializer = 'he_normal')(pool_2)
    conv_3 = BatchNormalization()(conv_3)
    conv_3 = Activation("relu")(conv_3)

    pool_3, mask_3 = MaxPoolingWithArgmax2D(pool_size=(2, 2))(conv_3)
    
    conv_4 = Conv2D(4*numFilters, (3,3), padding="same", kernel_initializer = 'he_normal')(pool_3)
    conv_4 = BatchNormalization()(conv_4)
    conv_4 = Activation("relu")(conv_4)

    pool_4, mask_4 = MaxPoolingWithArgmax2D(pool_size=(2, 2))(conv_4)
             
    unpool_1 = MaxUnpooling2D(pool_size=(2, 2))([pool_4, mask_4])

    conv_5 = Conv2D(2*numFilters, (3,3), padding="same", kernel_initializer = 'he_normal')(unpool_1)
    conv_5 = BatchNormalization()(conv_5)
    conv_5 = Activation("relu")(conv_5)
    
    unpool_2 =  MaxUnpooling2D(pool_size=(2, 2))([conv_5, mask_3])
    
    conv_6 = Conv2D(2*numFilters, (3,3), padding="same", kernel_initializer = 'he_normal')(unpool_2)
    conv_6 = BatchNormalization()(conv_6)
    conv_6 = Activation("relu")(conv_6)
    
    unpool_3 =  MaxUnpooling2D(pool_size=(2, 2))([conv_6, mask_2])
    
    conv_7 = Conv2D(numFilters, (3,3), padding="same", kernel_initializer = 'he_normal')(unpool_3)
    conv_7 = BatchNormalization()(conv_7)
    conv_7 = Activation("relu")(conv_7)
    
    unpool_4 =  MaxUnpooling2D(pool_size=(2, 2))([conv_7, mask_1])

    conv_8 = Conv2D(n_labels, (1, 1), padding="same",kernel_initializer = 'he_normal' )(unpool_4)
    conv_8 = BatchNormalization()(conv_8)
    outputs = Activation(output_mode)(conv_8)
    outputs = Cropping2D(cropping=((16, 16), (16, 16)))(outputs)


    segnet = Model(inputs=inputs, outputs=outputs)
    segnet.compile(optimizer=Nadam(lr=1e-3), loss=jaccard_coef_loss, metrics=['accuracy', jaccard_coef_int])

    return segnet

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
                sum_ = np.sum(y_batch)
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