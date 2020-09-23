import os
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.backend import binary_crossentropy
import tensorflow.keras.backend as K
import datetime
import h5py
import numpy as np
import pandas as pd

from createSegNet import *

def main():
	data_path = '../../../data'
	now = datetime.datetime.now()

	print('[{}] Creating and compiling model...'.format(str(datetime.datetime.now())))

	n_label = 3
	# model = createSegNet((96,96, 16), n_label, numFilters=32, output_mode='sigmoid')
	model = segnet_basic(orig_depth=64)

	print('[{}] Reading train...'.format(str(datetime.datetime.now())))
	f = h5py.File(os.path.join(data_path, 'train_16.h5'), 'r')

	images_list = [8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] # road images
	X_train = np.array(f['train'])[images_list,:,:,:]

	num_test_images = 3
    

	y_train = np.array(f['train_mask'])[:, [0,2]][images_list,:,:,:]
	y_train = np.concatenate([np.zeros((y_train.shape[0],1,y_train.shape[2], y_train.shape[3])), y_train], axis=1)

	print(y_train.shape)

	train_ids = np.array(f['train_ids'])
	f.close()

	batch_size = 128
	nb_epoch = 30
	suffix = 'roads_1_builds_3_'+str(datetime.datetime.now())[:16]

	callbackPath = "../../cache/SegNet/Checkpoints/{}_{}_{}".format(batch_size, nb_epoch, suffix)
	os.makedirs(callbackPath, exist_ok=True)

	history = keras.callbacks.History()
	callbacks = [
		history,
		keras.callbacks.ModelCheckpoint(callbackPath+"/weights.{epoch:02d}_{acc:.4f}_{val_acc:.4f}.h5",
		 save_best_only=False,
		 monitor='loss',
		 mode='min',
		 save_weights_only=True)
	]

	# shuffle images
	import random
	random.seed(a=1)
	mylist = random.sample(range(len(images_list)), len(images_list))

	X_train = X_train[mylist,:,:,:]
	y_train = y_train[mylist,:,:,:]

	# instantiate generators for train and validation
	trainGen = MyGenerator(X_train[:len(images_list)-num_test_images], y_train[:len(images_list)-num_test_images], batch_size, horizontal_flip=True, vertical_flip=True, swap_axis=True, min_true_label=20)
	testGen  = MyGenerator(X_train[len(images_list)-num_test_images:], y_train[len(images_list)-num_test_images:], batch_size, horizontal_flip=True, vertical_flip=True, swap_axis=True, min_true_label=20)

	model.fit_generator(trainGen, epochs=nb_epoch, steps_per_epoch=batch_size, validation_data=testGen, callbacks=callbacks, workers=16)

	# save important outputs
	save_model(model, "{batch}_{epoch}_{suffix}".format(batch=batch_size, epoch=nb_epoch, suffix=suffix))
	save_history(history, suffix)

	return

main()
