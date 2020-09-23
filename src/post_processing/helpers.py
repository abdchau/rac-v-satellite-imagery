import numpy as np
import skimage
import matplotlib.pyplot as plt
import skimage.feature as feature
import skimage.measure
import skimage.io
from math import atan2, degrees, sqrt, acos
from PIL import Image
from PIL import ImageDraw
from matplotlib import style
# style.use('dark_background')

def getAngle(a, b, c):
    angle = degrees(atan2(c[1]-b[1], c[0]-b[0]) - atan2(a[1]-b[1], a[0]-b[0]))
    return angle + 360 if angle < 0 else angle

def get_angle(point1, point2, point3):
	return atan2(point2[0] - point1[0], point2[1] - point1[1]) - atan2(point3[0] - point2[0], point3[1] - point2[1])

def getCorners(cnts, SKIP=26):

	corners = []
	for j in range(len(cnts)):
		if len(cnts[j]) > SKIP * 4:
			prev_x, prev_y = cnts[j][0][0], cnts[j][0][1]
			curr_x, curr_y = cnts[j][SKIP - 1][0], cnts[j][SKIP - 1][1]

		corner = []
		for i in range(2 * SKIP, len(cnts[j]), SKIP):
			if len(cnts[j]) < SKIP * 4:
				break
			next_x = cnts[j][i][0]
			next_y = cnts[j][i][1]

			if abs(get_angle((prev_x, prev_y), (curr_x, curr_y), (next_x, next_y))) > 0.25:
				corner.append(i-SKIP)

			prev_x, prev_y = curr_x, curr_y
			curr_x, curr_y = next_x, next_y

		corners.append(np.array(corner))
			
	return np.array(corners)

def drawEdges(image, cnts, corners=None):
	# cnts = skimage.measure.find_contours(image, 0.5, fully_connected='low', positive_orientation='low')
	img = Image.new('P', (image.shape[1],image.shape[0]), color=255)

	draw = ImageDraw.Draw(img)

	for j in range(len(cnts)):
		for i in range(len(cnts[j]) - 1):
			if corners is not None:
				if not any(abs(i - corners[j]) < 8):
					draw.line((cnts[j][i,1], cnts[j][i,0], cnts[j][i+1,1], cnts[j][i+1,0]), fill=0, width=5)
			else:
				draw.line((cnts[j][i,1], cnts[j][i,0], cnts[j][i+1,1], cnts[j][i+1,0]), fill=0, width=5)

	return img

def len_segment(segment):
	return (segment[0][0]-segment[-1][0])**2 + (segment[0][1]-segment[-1][1])**2

def separate(lst, indices, step, minLength):
	if len(lst) == 0:
		return [lst]

	startIndex = 0
	result = []
	for i in indices:
		slice_ = lst[startIndex:i-step]
		
		if len(slice_) != 0 and len_segment(slice_) > minLength**2:
			result.append(slice_)
		startIndex = i + step

	result.append(lst[startIndex:])
	return result

def separateLists(lsts, indices, step=8, minLength=20):
	if indices == False:
		return lsts
	result = []

	for i in range(len(lsts)):
		lst = separate(lsts[i], indices[i], step, minLength)
		for l in lst:
			# result.append(np.array([l[0], l[-1]]))
			result.append(l)

	return result

def dist(point1, point2):
	return sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def perp_distance(line1, line2, point):
	return np.linalg.norm(np.cross(line2-line1, line1-point))/np.linalg.norm(line2-line1)


def is_relevant(line1, line2, point):
	points = np.array([line1, line2, point])					# A, B, C
	lengths = [dist(points[i], points[i-1]) for i in range(3)]	# AC, AB, BC
	angles = []													# B, A

	if any(np.array(lengths)==0):
		# print(lengths)
		return False
	# print(lengths)
	for i in range(2):
		angles.append(degrees(acos(np.clip((lengths[i] * lengths[i] + lengths[i+1] * lengths[i+1] - lengths[i-1] * lengths[i-1])
			/ (2.0 * lengths[i] * lengths[i+1]), -1,1))))

	# print(angles)
	# plt.plot(points[:,0], points[:,1])
	# plt.show()

	return not any(np.array(angles)>90)

def is_dangerous(line1, line2, point, safe_distance=100):
	if not is_relevant(line1, line2, point):
		return False
	elif perp_distance(line1, line2, point) < safe_distance and perp_distance(line1,line2,point)>10:
		return True
	else:
		return False
