"""Plan for image analysis:
    1. Convert to grayscale
    2. Blob detection? Laplacian of Gaussian?
    3. Need to isolate the galactic core coordinates (brightest point of image?)"""

# imports

import cv2
from skimage import feature, color, io
import matplotlib.pyplot as plt
import numpy as np

# loading images

m31_img=cv2.imread("m31_20260904_024532_0_d7jusq_l.png")
m32_img=cv2.imread("m32_20250611_032223_2_qc9hiu_lrgb.png")
m110_img=cv2.imread("m110.jpg")

# cv2.imshow("m31",m31_img)
# cv2.imshow("m32",m32_img)
# cv2.imshow("m110",m110_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# blob detection (??)

def simple_blog_detection(): # yeah the built in simple blog detection really sucks
    params=cv2.SimpleBlobDetector_Params()
    params.filterByArea=True
    params.minArea=10
    params.filterByCircularity=False
    params.filterByConvexity=False
    params.filterByInertia=False

    detector=cv2.SimpleBlobDetector_create(params)

    blobs=detector.detect(m31_img)
    m31_img_blobs=cv2.drawKeypoints(m31_img,blobs,np.array([]),(0,0,255),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.imshow('Simple Blob Detection',m31_img_blobs)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def laplacian_of_gaussian_blog_detection(): # you can run this to check but it works pretty horrendously
    l=[]

    blobs_log=feature.blob_log(m31_img,max_sigma=30,num_sigma=10,threshold=0.1)
    blobs_log[:,2]*=(2**0.5) # accounting for blob radii

    print(blobs_log.shape)

    fig,ax=plt.subplots()
    ax.imshow(m31_img,cmap='gray')

    # need (x,y) in terms of a list of tuples, not as 2d array. Take the first two elements of each array and put them in a tuple.


    for i in range(len(blobs_log)):
        l.append((blobs_log[i][0],blobs_log[i][1]))

    print(l)

    for j in range(len(blobs_log)):
        r=blobs_log[j][2]
        patch=plt.Circle(l[j],r,color='red',linewidth=2,fill=False)
        ax.add_patch(patch)
    plt.show()