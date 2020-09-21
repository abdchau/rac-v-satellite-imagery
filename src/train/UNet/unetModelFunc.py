import os
import tensorflow as tf
import tensorflow.keras as keras
from units import *
import tensorflow.keras.backend as K
import pandas as pd
import numpy as np
import threading
#tf.enable_eager_execution()abd
from tensorflow.python.keras.utils.data_utils import Sequence
import random

smooth = 1e-12
num_channels = 16
num_mask_channels = 1
img_rows = 96
img_cols = 96
cache_path = '../../cache'


#======================================================================================================
#                                       THE MODEL ITSELF
#======================================================================================================


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



def UNetModel(shape=(512,512,1), origDepth=64):
	inputs = keras.layers.Input(shape=shape)
	
	# contraction block                                                          # input shapes
	contract, concat1 = contraction_unit(depth=origDepth*1)(inputs)              # 512x512
	contract, concat2 = contraction_unit(depth=origDepth*2)(contract)            # 256x256
	contract, concat3 = contraction_unit(depth=origDepth*4)(contract)            # 128x128
	contract, concat4 = contraction_unit(depth=origDepth*8)(contract)            # 64x64
	

	# expansion block
	expand = bottomExpansionUnit(depth=origDepth*16, dropout=0.5)(contract)      # 32x32
	expand = expansion_unit(depth=origDepth*8, dropout=0.5)([expand, concat4])   # 64x64
	expand = expansion_unit(depth=origDepth*4, dropout=0.5)([expand, concat3])   # 128x128
	expand = expansion_unit(depth=origDepth*2, dropout=0.5)([expand, concat2])   # 256x256

	# output
	expand = expansion_unit(depth=origDepth, up=False)([expand, concat1])        # 512x512
	output = Conv2D(1, 1, activation='sigmoid', kernel_initializer='he_normal')(expand)
	output = Cropping2D(cropping=((16, 16), (16, 16)))(output)
	# output = keras.layers.Softmax()(output)

	unet = keras.models.Model(inputs=inputs, outputs=output)
	unet.compile(optimizer=keras.optimizers.Nadam(lr=1e-3), loss=jaccard_coef_loss, metrics=['accuracy', jaccard_coef_int])
	# Unet.summary()

	return unet

#======================================================================================================
#                                       MISCELLANEOUS HELPERS
#======================================================================================================

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
	pd.DataFrame(history.history).to_csv(filename, index=False)

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