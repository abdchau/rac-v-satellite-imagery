from helpers import *


def contours():
	# Reading image 
	# img2 = cv2.imread('road.jpg', cv2.IMREAD_COLOR) 
	# img = skimage.io.imread("road.png", as_gray=True)
	with open("prediction.npy", "rb") as f:
		img = np.load(f)

	cnts = skimage.measure.find_contours(np.asarray(img), 0.5, fully_connected='low', positive_orientation='low')

	possible_road_segments = separateLists(cnts, getCorners(cnts, show=False), minLength=80)
	image1 = drawEdges(img, possible_road_segments)
	np.save('all_segments.npy', image1)
	# image1.show()

	max_index = 0
	max_length = 0

	for i, segment in enumerate(possible_road_segments):
		length = len_segment(segment)
		if length > max_length:
			max_index = i
			max_length = length

	longest_segment = possible_road_segments[max_index]
	image2 = drawEdges(img, [longest_segment])
	np.save('best_segment.npy', image2)
	# image2.show()

	angles = []
	for segment in possible_road_segments:
		deltaX = segment[0][1] - segment[-1][1]
		deltaY = segment[0][0] - segment[-1][0]

		angles.append(int(degrees(atan2(deltaY, deltaX))+90)%180)
		# plt.plot(segment[:,0], segment[:,1], linewidth=1)

	# plt.legend(angles)
	# plt.show()
	print("Best segment length:", sqrt(max_length), "Angle:", angles[max_index], "degrees")

if __name__ == "__main__":
	contours()