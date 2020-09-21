import os
import tensorflow as tf
import tensorflow.keras as keras
from units import *
from tensorflow.keras.backend import binary_crossentropy
import tensorflow.keras.backend as K
import datetime
import h5py
import os
import numpy as np
import pandas as pd
#tf.enable_eager_execution()abd
from tensorflow.python.keras.utils.data_utils import Sequence


from unetModelFunc import *

def main():
	data_path = '../../../data'
	CROSS = '128_3_roads_3_2020-07-14 22:14'

	now = datetime.datetime.now()

	print('[{}] Creating and compiling model...'.format(str(datetime.datetime.now())))

	model = read_model(cross=CROSS, load=True)
	# model = UNetModel(shape=(96,96, 16), origDepth=32)
	# model.load_weights(os.path.join('./cache/Checkpoints/128_21_roads_3_2020-07-07 17:58', 'weights_19_0.9835_0.1518.h5'))
	# model.load_weights(os.path.join('./cache/Checkpoints/128_21_roads_3_2020-07-08 03:38', 'weights_21_0.9839_0.2210.h5'))
	model.compile(optimizer=keras.optimizers.Nadam(lr=1e-4), loss=jaccard_coef_loss, metrics=['accuracy', jaccard_coef_int])

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
	nb_epoch = 2
	suffix = 'roads_3_'+str(datetime.datetime.now())[:16]

	callbackPath = "../../cache/UNet/Checkpoints/{}_{}_{}".format(batch_size, nb_epoch, suffix)
	os.makedirs(callbackPath, exist_ok=True)

	history = keras.callbacks.History()
	callbacks = [
		history,
		keras.callbacks.ModelCheckpoint(callbackPath+"/weights_{epoch:02d}_{jaccard_coef_int:.4f}_{val_jaccard_coef_int:.4f}.h5",
		 save_best_only=False,
		 monitor='val_jaccard_coef_int',
		 mode='max',
		 save_weights_only=True)
	]
	#model.compile(optimizer=Nadam(lr=1e-3), loss=jaccard_coef_loss, metrics=['binary_crossentropy', jaccard_coef_int])

	import random
	random.seed(a=1)
	mylist = random.sample(range(len(images_list)), len(images_list))

	X_train = X_train[mylist,:,:,:]
	y_train = y_train[mylist,:,:,:]

	trainGen = MyGenerator(X_train[:len(images_list)-num_test_images], y_train[:len(images_list)-num_test_images], batch_size, horizontal_flip=True, vertical_flip=True, swap_axis=True, min_true_label=100)
	testGen  = MyGenerator(X_train[len(images_list)-num_test_images:], y_train[len(images_list)-num_test_images:], batch_size, horizontal_flip=True, vertical_flip=True, swap_axis=True, min_true_label=500)

	model.fit_generator(trainGen, epochs=nb_epoch, steps_per_epoch=batch_size, validation_data=testGen, callbacks=callbacks, workers=8)

	#model.fit(X_train, y_train, batch_size=batch_size, epochs=nb_epoch)
	save_model(model, "{batch}_{epoch}_{suffix}_continued".format(batch=batch_size, epoch=nb_epoch, suffix=suffix))
	append_history(history, suffix)

	return

main()
