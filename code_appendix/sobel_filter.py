import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('Lenna_img.png',0)



# blur = cv2.GaussianBlur(img,(5,5),0) gaussian blur
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)




plt.subplot(131),plt.imshow(img, 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(132),plt.imshow(sobelx, 'gray')
plt.title('Sobel X direction filter'), plt.xticks([]), plt.yticks([])
plt.subplot(133),plt.imshow(sobelx, 'gray')
plt.title('Sobel Y direction Filter'), plt.xticks([]), plt.yticks([])

plt.show()