import os
import math
from numpy import ma
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from orientation_utils import getKeypoints, getVectorMapsAngles

image_path = '/home/chaurasi/cs433/Project2/data/training/images/satImage_002.png'
gt_path = '/home/chaurasi/cs433/Project2/data/training/groundtruth/satImage_002.png'

os.makedirs('verification_plots', exist_ok = True)

# Function to plot vector map to check proper working of vector map generation
def plotVecMap(vectorMap, image, figsize_=(6,6), name=''):
    fig = plt.figure(figsize=figsize_)
    plt.axis('off')
    plt.imshow(image)

    U = vectorMap[:,:,0] * -1
    V = vectorMap[:,:,1]
    X, Y = np.meshgrid(np.arange(U.shape[0]), np.arange(U.shape[1]))
    M = np.zeros(U.shape, dtype='bool')
    M[U**2 + V**2 < 0.5 * 0.5] = True
    U = ma.masked_array(U, mask=M)
    V = ma.masked_array(V, mask=M)

    s = 15
    Q = plt.quiver(X[::s,::s], Y[::s,::s], U[::s,::s], V[::s,::s],scale=70,headaxislength=3,headwidth=4,
                   width=0.008,alpha=.8, color='y')
    fig = matplotlib.pyplot.gcf()
    plt.savefig('verification_plots/verification_vector_map.png', bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.show()

# Generate and plot vector map and angle map
gt_plot = cv2.imread(gt_path, 0).astype(np.float32)/ 255.0
keypoints = getKeypoints(
        gt_plot, is_gaussian=False, smooth_dist=4
    )

vecmap, vecmap_angles = getVectorMapsAngles(
        (int(math.ceil(400 / (1 * 1.0))), int(math.ceil(400 / (1 * 1.0)))), keypoints, theta=10, bin_size=10
    )
plotVecMap(vecmap, cv2.imread(image_path).astype(np.uint8))

plt.axis('off')
plt.title("Orientation Angles Vector Map")
plt.imshow(vecmap_angles, vmin=0, vmax=36)
plt.colorbar()
plt.savefig('verification_plots/verification_orientation_angle_map.png', bbox_inches='tight', pad_inches=0.1, dpi=300)
plt.show()