#import numpy as np
#import os
#from tqdm import tqdm_notebook as tqdm
#import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Dropout, UpSampling2D, Concatenate, MaxPool2D, Softmax, Activation, Cropping2D

def contraction_unit(depth=64, dropout=0.5):
	#conv, conv, batchnorm, dropout, maxpool
		
	def contraction(inputs):
		conv = Conv2D(depth, 3, activation='elu', kernel_initializer = 'he_normal', padding='same')(inputs)
		conv = Conv2D(depth, 3, kernel_initializer = 'he_normal', padding='same')(conv) #this will be concatenated
		conv = BatchNormalization()(conv)
		conv = Activation(activation='elu')(conv)

		drop = Dropout(dropout)(conv)
		drop = MaxPool2D(pool_size=2, padding='same')(drop)

		return drop, conv
	return contraction


def expansion_unit(depth=1024, dropout=None, up=True):
	#concat, conv, conv, upsample, batchnorm, dropout

	def expansion(concat):
		concat = Concatenate(axis=3)(concat)

		conv = Conv2D(depth, 3, activation='elu', kernel_initializer='he_normal', padding='same')(concat)
		conv = Conv2D(depth, 3, kernel_initializer='he_normal', padding='same')(conv)
		conv = BatchNormalization()(conv)
		conv = Activation(activation='elu')(conv)

		if up:
			drop = UpSampling2D(size=2)(conv)
		else:
			drop = conv
		
		if dropout != None:
			drop = Dropout(dropout)(drop)

		return drop
	return expansion

def bottomExpansionUnit(depth=1024, dropout=None):
	#conv, conv, upsample, batchnorm, dropout

	def expansion(inputs):
		conv = Conv2D(depth, 3, activation='elu', kernel_initializer='he_normal', padding='same')(inputs)
		conv = Conv2D(depth, 3, kernel_initializer='he_normal', padding='same')(conv)
		conv = BatchNormalization()(conv)
		conv = Activation(activation='elu')(conv)
		conv = UpSampling2D(size=2)(conv)
		
		if dropout != None:
			conv = Dropout(dropout)(conv)

		return conv
	return expansion