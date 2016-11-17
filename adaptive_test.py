import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('Lenna_img.png',0)
img_smoothed = cv2.medianBlur(img,5)

ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)

edges = cv2.Canny(img,100,200)
th2 =  np.invert(edges)
th3 = cv2.adaptiveThreshold(img_smoothed,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)
th4 = np.invert(cv2.Canny(th3,100,200))
cv2.imwrite('edge_blur.png',th2)
cv2.imwrite('gaussian_blur.png',th3)

titles = ['Original Image', 'Global Thresholding (v = 127)',
            'Edge Detection', 'Adaptive Gaussian Thresholding', 'Gauss + Edge']
images = [img, th1, th2, th3, th4]

for i in xrange(5):
    plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()