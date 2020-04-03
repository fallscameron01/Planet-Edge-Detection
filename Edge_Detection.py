"""
This file contains PlanetDetection, which will detect the edges of a planet in an image.
"""

import numpy as np
from scipy.ndimage import generic_filter, convolve
import cv2


def PlanetDetection(imagepath):
    """
    PlanetDetection will detect the edges of a planet in an image.

    Parameters
    ----------
    imagepath - string - the path of the image

    Return
    ------
    ndarray - edges image
    """
    # Read Image
    image = cv2.imread(imagepath)

    # K Means Clustering
    vec = np.float32(image.reshape((-1, 3)))

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
    
    return edges


if __name__ == "__main__":
    imgpath = 'images/Mars.jpg'
    edges = PlanetDetection(imgpath)
    
    cv2.imshow("Edges", edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()