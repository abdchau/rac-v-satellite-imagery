import os
import tensorflow as tf
import tensorflow.keras as keras
from units import *
from tensorflow.keras.backend import binary_crossentropy
import tensorflow.keras.backend as K
import datetime
import h5py
import numpy as np
import pandas as pd
from tensorflow.python.keras.utils.data_utils import Sequence

from unetModelFunc import *

def main():
	data_path = '../../../data'
	now = datetime.datetime.now()

	print('[{}] Creating and compiling model...'.format(str(datetime.datetime.now())))

	model = UNetModel(shape=(96,96, 16), origDepth=32)

	print('[{}] Reading train...'.format(str(datetime.datetime.now())))
	f = h5py.File(os.path.join(data_path, 'train_16.h5'), 'r')

	X_train = np.array(f['train'])

	images_list = [8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] # road images
	# images_list = list(range(25))
	num_test_images = 3
    
	y_train = np.array(f['train_mask'])[:, 2]
	y_train = np.expand_dims(y_train, 1)

	X_train = X_train[images_list,:,:,:]
	y_train = y_train[images_list,:,:,:]
	print(y_train.shape)

	train_ids = np.array(f['train_ids'])
	f.close()

	batch_size = 128
	nb_epoch = 3
	suffix = 'roads_3_'+str(datetime.datetime.now())[:16]

	callbackPath = "../../cache/UNet/Checkpoints/{}_{}_{}".format(batch_size, nb_epoch, suffix)
	os.makedirs(callbackPath, exist_ok=True)

	history = keras.callbacks.History()
	callbacks = [
		history,
		keras.callbacks.ModelCheckpoint(callbackPath+"/weights.{epoch:02d}_{jaccard_coef_int:.4f}_{val_jaccard_coef_int:.4f}.h5",
		 save_best_only=False,
		 monitor='val_jaccard_coef_int',
		 mode='max',
		 save_weights_only=True)
	]

	# shuffle images
	import random
	random.seed(a=1)
	mylist = random.sample(range(len(images_list)), len(images_list))

	X_train = X_train[mylist,:,:,:]
	y_train = y_train[mylist,:,:,:]

	# instantiate generators for train and validation
	trainGen = MyGenerator(X_train[:len(images_list)-num_test_images], y_train[:len(images_list)-num_test_images], batch_size, horizontal_flip=True, vertical_flip=True, swap_axis=True)
	testGen  = MyGenerator(X_train[len(images_list)-num_test_images:], y_train[len(images_list)-num_test_images:], batch_size, horizontal_flip=True, vertical_flip=True, swap_axis=True)

	model.fit_generator(trainGen, epochs=nb_epoch, steps_per_epoch=batch_size, validation_data=testGen, callbacks=callbacks, workers=16)

	# save important outputs
	save_model(model, "{batch}_{epoch}_{suffix}".format(batch=batch_size, epoch=nb_epoch, suffix=suffix))
	save_history(history, suffix)

	return

main()
