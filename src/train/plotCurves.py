import matplotlib.pyplot as plt
import csv

with open('history_road_3_.csv') as csv_file:
	csv_reader = csv.reader(csv_file, delimiter=',')
	line_count = 0

	loss = []
	acc = []
	jaccard_coef_int = []

	test_loss = []
	test_acc = []
	test_jaccard_coef_int = []

	for row in csv_reader:
		if line_count != 0:
			loss.append(float(row[0]))
			acc.append(float(row[1]))
			jaccard_coef_int.append(float(row[2]))

			test_loss.append(float(row[3]))
			test_acc.append(float(row[4]))
			test_jaccard_coef_int.append(float(row[5]))

		line_count+=1

	print(f'Processed {line_count} lines.')
	epochs = list(range(1, line_count))

	fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(16,8))
	ax[0][0].plot(epochs, loss)
	ax[1][0].plot(epochs, test_loss)

	ax[0][1].plot(epochs, acc)
	ax[1][1].plot(epochs, test_acc)

	ax[0][2].plot(epochs, jaccard_coef_int)
	ax[1][2].plot(epochs, test_jaccard_coef_int)
	

	ax[0][0].legend(['loss'])
	ax[0][1].legend(['acc'])
	ax[0][2].legend(['jaccard_coef_int'])

	ax[1][0].legend(['test_loss'])
	ax[1][1].legend(['test_acc'])
	ax[1][2].legend(['test_jaccard_coef_int'])
	
	ax[0][0].title.set_text('Train Loss')
	ax[0][1].title.set_text('Train Accuracy')
	ax[0][2].title.set_text('Train Intersection Over Union')

	ax[1][0].title.set_text('Test Loss')
	ax[1][1].title.set_text('Test Accuracy')
	ax[1][2].title.set_text('Test Intersection Over Union')

	fig.canvas.set_window_title('Behold')

	plt.show()