import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('Lenna_img.png',0)
edges = cv2.Canny(img,100,100)
inverted_edges = np.invert(edges)
cv2.imwrite('edge.png',inverted_edges)
plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(inverted_edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()