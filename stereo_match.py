import numpy as np
import cv2
from matplotlib import pyplot as plt
imgL = cv2.imread('left1.jpg',0)
imgR = cv2.imread('right1.jpg',0)
#stereo = cv2.StereoBM_create(numDisparities=32, blockSize=33)

stereo = cv2.StereoSGBM_create(minDisparity = 0,
        numDisparities = 64, #num_disp,
        blockSize = 5, #16,
        P1 = 200,
        P2 = 400,
        disp12MaxDiff = 1, #1,
        uniquenessRatio = 0,
        speckleWindowSize = 300,
        speckleRange = 7
)

disparity = stereo.compute(imgL, imgR) #.astype(np.float32) / 16.0
#disparity = stereo.compute(imgL,imgR)
plt.imshow(disparity,'gray')
plt.show()
