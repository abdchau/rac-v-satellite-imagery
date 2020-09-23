from helpers import *
# from tensorflow import one_hot, argmax


def get_segments(img, cornerify=True, minLength=900):
	# Reading image 
	# img2 = cv2.imread('road.jpg', cv2.IMREAD_COLOR) 
	# img = skimage.io.imread("road.png", as_gray=True)
	cnts = skimage.measure.find_contours(np.asarray(img), 0.5, fully_connected='low', positive_orientation='low')
	possible_road_segments = separateLists(cnts, getCorners(cnts) if cornerify else False, minLength=minLength)
	return possible_road_segments

def get_longest_segment(possible_road_segments):
	max_index = 0
	max_length = 0

	for i, segment in enumerate(possible_road_segments):
		length = len_segment(segment)
		if length > max_length:
			max_index = i
			max_length = length

	longest_segment = possible_road_segments[max_index]

	angles = []
	for segment in possible_road_segments:
		deltaX = segment[0][1] - segment[-1][1]
		deltaY = segment[0][0] - segment[-1][0]

		angles.append(int(degrees(atan2(deltaY, deltaX))+90)%180)
		# plt.plot(segment[:,0], segment[:,1], linewidth=1)

	# plt.legend(angles)
	# plt.show()
	return longest_segment, angles[max_index], sqrt(max_length)

def get_safest_segment(img):
	possible_road_segments = get_segments(img[:,:,1])
	np.save('all_segments.npy', possible_road_segments)
	possible_road_segments=list(possible_road_segments)
	building_segments = list(get_segments(img[:,:,2], cornerify=False, minLength=225))

	delete = []
	print("Total road segments found:", len(possible_road_segments))
	SKIP = 20
	for j in range(len(possible_road_segments)):
		for build_segment in building_segments:
			if j in delete:
				continue
			for i in range(0, len(build_segment), SKIP):
				if is_dangerous(possible_road_segments[j][0],possible_road_segments[j][-1],build_segment[i],safe_distance=20):
					delete.append(j)
					break

	for i in delete[::-1]:
		del possible_road_segments[i]

	print("Road segments at a safe distance from buildings:", len(possible_road_segments))
	longest_segment, angle, length = get_longest_segment(possible_road_segments)
	print("Best segment length: ", length, "Angle: ", angle, "degrees")
	return longest_segment


if __name__=='__main__':
	with open("prediction.npy", "rb") as f:
		img = np.load(f)

	# img = one_hot(argmax(np.round(one_hot(img, depth=3).numpy()), axis=-1).numpy(), depth=3).numpy()
	longest_segment = get_safest_segment(img)
	image1 = drawEdges(img, [longest_segment])
	np.save('best_segment.npy',np.array(image1))