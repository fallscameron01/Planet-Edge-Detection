import numpy as np

import matplotlib.pyplot as plt

from scipy.ndimage import generic_filter, convolve

import cv2

# Initial Image
image = cv2.imread('images/Mars.jpg')

# K Means Clustering
vec = image.reshape((-1, 3))
vec = np.float32(vec)

termination_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 2
_, label, center = cv2.kmeans(vec, K, bestLabels=None, criteria=termination_criteria, attempts=10, flags=0)

meansImg = np.uint8(center)[label.flatten()]
meansImg = meansImg.reshape((image.shape))

# Edge Detection
edge_kernel = np.array((
    [-1, -1, -1], 
    [-1, 8, -1], 
    [-1, -1, -1]))

grayscaleImg = np.mean(meansImg, axis=2)

edges = convolve(grayscaleImg, edge_kernel)
binarized_edges = np.where(edges > .25, 1, 0)

cv2.imshow("Edges", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()