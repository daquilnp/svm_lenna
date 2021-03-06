import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('Lenna_img.png',0)



blur = cv2.GaussianBlur(img,(5,5),0) 


plt.subplot(121),plt.imshow(img, 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(blur, 'gray')
plt.title('Gaussian Filter'), plt.xticks([]), plt.yticks([])

plt.show()