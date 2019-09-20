import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from scipy.ndimage import generic_filter, convolve
import cv2

# Initial Image
image = mpimg.imread('images/Mars.jpg')
plt.imshow(image)
plt.show()

# K Means Clustering
vec = image.reshape((-1, 3))
vec = np.float32(vec)

termination_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

K = 2
_, label, center = cv2.kmeans(vec, K, bestLabels=None, criteria=termination_criteria, attempts=10, flags=0)

meansImg = np.uint8(center)[label.flatten()]
meansImg = meansImg.reshape((image.shape))

plt.imshow(meansImg)
plt.show()

# Edge Detection
edge_kernel = [
    [-1, -1, -1], 
    [-1, 8, -1], 
    [-1, -1, -1]]

edge_kernel = np.array(edge_kernel)

grayscaleImg = np.mean(meansImg, axis=2)
plt.imshow(grayscaleImg, cmap='gray')
plt.show()

edges = convolve(grayscaleImg, edge_kernel)
binarized_edges = np.where(edges > .25, 1, 0)

plt.imshow(binarized_edges, cmap='gray')
plt.show()