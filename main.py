import matplotlib.pyplot as plt 
from sklearn import datasets
from sklearn import svm
import cv2
import numpy as np
from random import randint
import time
import collections
TRAINING_SET_SIZE = 20000
TESTING_SET_SIZE = 20000

def main():
	# img = cv2.imread('Lenna_img.png',0)

	# img_bin = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
	#             cv2.THRESH_BINARY,11,2)
	img = cv2.imread('Lenna_img.png', cv2.CV_LOAD_IMAGE_GRAYSCALE)
	thresh = 135
	img_bin = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)[1]
	overlap_image = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)[1]

	training_set = GetRandomSet(img_bin, overlap_image, TRAINING_SET_SIZE)
	testing_set = GetRandomSet(img_bin, overlap_image, TESTING_SET_SIZE)

	svm_clf = TrainNeuralNetwork(training_set)

	print TestNeuralNetwork(testing_set, svm_clf)

	plt.subplot(121),plt.imshow(img)
	plt.title('Original Image'), plt.xticks([]), plt.yticks([])
	plt.subplot(122),plt.imshow(img_bin, cmap=plt.cm.gray_r)
	plt.title('Binary'), plt.xticks([]), plt.yticks([])
	plt.show()


	
def GetRandomSet(img_bin, overlap_image, set_size):

	x_range,y_range = img_bin.shape
	random_data_set = [[],[],[]]
	i = 0
	while i < set_size:
		data_point = []
		data_point_coordinate = [0]*2
		data_point_input = 0
		data_point_expected_output = 0
		
		flag = 0
		while flag == 0:
			data_point_coordinate[0] = randint(0, x_range-1)
			data_point_coordinate[1] = randint(0, y_range-1)
			# print "overlap: " + str(overlap_image[data_point_coordinate[0], data_point_coordinate[1]])
			if overlap_image[data_point_coordinate[0], data_point_coordinate[1]] != 150:
				overlap_image[data_point_coordinate[0], data_point_coordinate[1]] = 150
				flag = 1


		data_point_input = img_bin[data_point_coordinate[0], data_point_coordinate[1]]
		# print "binary: " + str(data_point_input)
		if data_point_input == 0:
			data_point_expected_output = 0 #not lenna
		else:
			data_point_expected_output = 1 #lenna feature

		data_point = [data_point_coordinate, [data_point_input,1], data_point_expected_output]
		# if data_point not in random_data_set and data_point not in taken_values_set:
		random_data_set[0].append(data_point[0])
		random_data_set[1].append(data_point[1])
		random_data_set[2].append(data_point[2])
			# print str(i) + ": added new one"
		i+= 1

	return random_data_set


def TrainNeuralNetwork(training_set):
	nn_inputs = training_set[1]
	nn_expcted_outputs = training_set[2]
	clf = svm.SVC(gamma=0.001, C=100)
	clf.fit(nn_inputs, nn_expcted_outputs)

	return clf

def TestNeuralNetwork(testing_set, svm_clf):
	nn_inputs = testing_set[1]
	nn_expcted_outputs = testing_set[2]
	index = 0
	correct = 0
	for i in nn_inputs:
		if svm_clf.predict(i) == nn_expcted_outputs[index]:
			correct += 1
		index += 1
	return float(correct/len(testing_set[1]))

start_time = time.time()
main()
print str(time.time() - start_time)
# print clf.predict(digits.data[-2])
# plt.imshow(digits.images[3], cmap=plt.cm.gray_r, interpolation="nearest")
# plt.show()