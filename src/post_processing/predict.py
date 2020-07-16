import numpy as np
import matplotlib.pyplot as plt
import os
import h5py

from unetModelFunc import *

def show_images(images, cols = 1, titles = None):
	"""Display a list of images in a single figure with matplotlib.
	
	Parameters
	---------
	images: List of np.arrays compatible with plt.imshow.
	
	cols (Default = 1): Number of columns in figure (number of rows is 
						set to np.ceil(n_images/float(cols))).
	
	titles: List of titles corresponding to each image. Must have
			the same length as titles.
	"""
	assert((titles is None) or (len(images) == len(titles)))
	n_images = len(images)
	if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
	fig = plt.figure()
	for n, (image, title) in enumerate(zip(images, titles)):
		a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
		if image.ndim == 2:
			plt.gray()
		plt.imshow(image)
		a.set_title(title)
	fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
	plt.show()

def main():
	data_path = './Data'


	f = h5py.File(os.path.join(data_path, 'train_16.h5'), 'r')

	X_train = np.array(f['train'])

	y_train = np.array(f['train_mask'])[:, 0]
	y_train = np.expand_dims(y_train, 1)
	print(y_train.shape)

	import random
	random.seed(a=42)
	mylist = random.sample(range(25), 25)

	X_train = X_train[mylist,:,:,:]
	y_train = y_train[mylist,:,:,:]

	img = 9
	batch_size = 4
	tol = 6

	startWidth = 150
	endWidth = startWidth + 432
	startHeight = 220
	endHeight = startHeight + 304

	X_train = X_train[img]
	y_train = y_train[img]

	print(X_train.shape)
	print(type(X_train[1,1,1]))

	
	model = UNetModel(shape=(432,304,16), origDepth=32)
	#model.load_weights("./cache/Checkpoints/128_49_buildings_3_split_dingdong_continued/weights.38_0.7921_0.7022.h5")
	model.load_weights(os.path.join('../cache/Checkpoints/128_20_roads_3_2020-06-26 18:43', 'weights.20_0.8313_0.2293.h5'))

	#a = X_train.swapaxes(0,2)[1600:1696,100:196,:]

	moi = X_train[:,startHeight:endHeight, startWidth:endWidth]
	print(moi.shape)
	#preGen  = PredGenerator(moi, batch_size, tol=tol)

	#pred = model.predict_generator(preGen)

	finalPred = np.squeeze(model.predict(np.expand_dims(moi.swapaxes(0,2), axis=0)))


	'''
	plt.imshow(X_train[13:,:,:].swapaxes(0,2).astype(np.float64)[1600:1780,100:500,:])
	plt.show()
	plt.imshow(y_train.swapaxes(0,2).astype(np.float64)[:,:,0][1600:1780,100:500], cmap=plt.cm.binary)
	plt.show()
	plt.imshow(finalPred, cmap=plt.cm.binary)
	plt.show()
	'''
	show_images([X_train[13:,:,:].swapaxes(0,2).astype(np.float64)[startWidth+16:endWidth-16,startHeight+16:endHeight-16,:], 1-y_train.swapaxes(0,2).astype(np.float64)[:,:,0][startWidth+16:endWidth-16,startHeight+16:endHeight-16], 1-finalPred],titles=["Original Image", "Label", "Predicted"])

	print("o hai mark")

	f.close()
	return

main()