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
IMAGE_FILE = "Lenna_img.png"

def main():

	img_tuple = clean_image_edge()
	

	training_set = GetTrainingSet(img_tuple[0], img_tuple[1], TRAINING_SET_SIZE)
	testing_set = GetRandomSet(img_tuple[0], img_tuple[1], TESTING_SET_SIZE)

	svm_clf = TrainNeuralNetwork(training_set)

	# print TestNeuralNetwork(testing_set, svm_clf)
	result_img = GenerateImage(img_tuple[0], svm_clf)
	cv2.imwrite('result_img.png',result_img)
	plt.subplot(131),plt.imshow(cv2.imread(IMAGE_FILE, cv2.CV_LOAD_IMAGE_GRAYSCALE), cmap=plt.cm.gray_r)
	plt.title('Original Lenna Image'), plt.xticks([]), plt.yticks([])
	plt.subplot(132),plt.imshow(np.invert(img_tuple[0]), cmap=plt.cm.gray_r)
	plt.title('Canny Edge Detection Image'), plt.xticks([]), plt.yticks([])
	plt.subplot(133),plt.imshow(result_img, cmap=plt.cm.gray_r)
	plt.title('SVM Reconstructed Image'), plt.xticks([]), plt.yticks([])
	plt.show()

def clean_image_edge():
	img = cv2.imread(IMAGE_FILE,0)
	edges = cv2.Canny(img,100,200)
	inverted_edges = np.invert(edges)
	overlap_image = np.invert(edges)
	return [inverted_edges, overlap_image]

def clean_image_binary():
	img = cv2.imread(IMAGE_FILE, cv2.CV_LOAD_IMAGE_GRAYSCALE)
	thresh = 135
	img_bin = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)[1]
	overlap_image = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)[1]
	return [img_bin, overlap_image]

def clean_image_gaussian():
	img = cv2.imread(IMAGE_FILE, 0)
	img = cv2.medianBlur(img,5)
	img_bin = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)
	overlap_image = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)
	return [img_bin, overlap_image]
	
def GetTrainingSet(img_bin, overlap_image, set_size):
	x_range,y_range = img_bin.shape
	random_data_set = [[],[],[]]
	lennna_data_set = []
	not_lenna_data_set = []

	for x in range(x_range):
		for y in range(y_range):
			if (img_bin[x, y] == 255):
				lennna_data_set.append([x,y])

	for k in range(x_range):
		for l in range(y_range):
			if (img_bin[k, l] == 0):
				not_lenna_data_set.append([k,l])

	lenna_set_size = set_size - int((set_size)*2.0/3)
	not_lenna_set_size = set_size - lenna_set_size
	data_sets = [[lennna_data_set, lenna_set_size, 1], [not_lenna_data_set, not_lenna_set_size, 0]]
	for data_set in data_sets:

		i = 0
		while i < data_set[1]:
			data_point = []
			data_point_coordinate = [0]*2
			data_point_input = 0
			data_point_expected_output = 0
			
			flag = 0
			while flag == 0:

				coordinate = randint(0, len(data_set[0])-1)
				data_point_coordinate[0] = data_set[0][coordinate][0]
				data_point_coordinate[1] = data_set[0][coordinate][1]
				# print "overlap: " + str(overlap_image[data_point_coordinate[0], data_point_coordinate[1]])
				if overlap_image[data_point_coordinate[0], data_point_coordinate[1]] != 150:
					overlap_image[data_point_coordinate[0], data_point_coordinate[1]] = 150
					flag = 1


			data_point_input = img_bin[data_point_coordinate[0], data_point_coordinate[1]]
			# print "binary: " + str(data_point_input)

			data_point = [data_point_coordinate, [data_point_input,1], data_set[2]]
			# if data_point not in random_data_set and data_point not in taken_values_set:
			random_data_set[0].append(data_point[0])
			random_data_set[1].append(data_point[1])
			random_data_set[2].append(data_point[2])
				# print str(i) + ": added new one"
			i+= 1

	return random_data_set
	
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
	clf = svm.SVC(kernel='linear')
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

def GenerateImage(img, svm_clf):
	x_range,y_range = img.shape
	result_img = np.zeros([x_range,y_range,3],dtype=np.uint8)
	result_img.fill(255)
	for x in range(x_range):
		for y in range(y_range):
			if (svm_clf.predict([img[x, y],1]) == 0):
				result_img[x,y] = 0
				

	return result_img


start_time = time.time()
main()
print str(time.time() - start_time)










