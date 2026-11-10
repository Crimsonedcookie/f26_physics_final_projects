"""Plan for image analysis:
    1. Convert to grayscale
    2. Blob detection? Laplacian of Gaussian?
    3. Need to isolate the galactic core coordinates (the brightest point of image?)
    4. Calculate the radius at every point in an array
    5. Group coordinates by radius, average (?) their brightness and plot that graph"""

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

# def simple_blog_detection(): # yeah the built in simple blog detection really sucks
#     params=cv2.SimpleBlobDetector_Params()
#     params.filterByArea=True
#     params.minArea=10
#     params.filterByCircularity=False
#     params.filterByConvexity=False
#     params.filterByInertia=False
#
#     detector=cv2.SimpleBlobDetector_create(params)
#
#     blobs=detector.detect(m31_img)
#     m31_img_blobs=cv2.drawKeypoints(m31_img,blobs,np.array([]),(0,0,255),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#
#     cv2.imshow('Simple Blob Detection',m31_img_blobs)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
#
# def laplacian_of_gaussian_blog_detection(): # you can run this to check but it works pretty horrendously
#     l=[]
#
#     blobs_log=feature.blob_log(m31_img,max_sigma=30,num_sigma=10,threshold=0.1)
#     blobs_log[:,2]*=(2**0.5) # accounting for blob radii
#
#     print(blobs_log.shape)
#
#     fig,ax=plt.subplots()
#     ax.imshow(m31_img,cmap='gray')
#
#     # need (x,y) in terms of a list of tuples, not as 2d array. Take the first two elements of each array and put them in a tuple.
#
#
#     for i in range(len(blobs_log)):
#         l.append((blobs_log[i][0],blobs_log[i][1]))
#
#     print(l)
#
#     for j in range(len(blobs_log)):
#         r=blobs_log[j][2]
#         patch=plt.Circle(l[j],r,color='red',linewidth=2,fill=False)
#         ax.add_patch(patch)
#     plt.show()

# grayscaling
m31_gray=cv2.cvtColor(m31_img, cv2.COLOR_BGR2GRAY)
m32_gray=cv2.cvtColor(m32_img, cv2.COLOR_BGR2GRAY)
m110_gray=cv2.cvtColor(m110_img, cv2.COLOR_BGR2GRAY)

# Finding center of galaxy based on brightness, an approximation that isn't always true (thus a source of error)

def find_center(image, kernel_size, show_center):
    img_gaussian_blur = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # convert to grayscale
    img_gaussian_blur = cv2.GaussianBlur(img_gaussian_blur, kernel_size, 0)  # gaussian blur to account for noise
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(img_gaussian_blur)  # finding point of greatest intensity in image
    show_galactic_center = cv2.circle(img_gaussian_blur, maxLoc, 10, (255, 255, 0), 5)
    if show_center==True:
        cv2.imshow("grayscaled+gaussian blurred",show_galactic_center)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return maxLoc


suggested_m110_kernel=(21,21) # This is an EXTREMELY imperfect method to finding the centers of the galaxies.
suggested_m31_kernel=(5,5) # Due to time constraints I can't spend too long on this so for now it works
suggested_m32_kernel=(37,37) # However, it is a source of error to account for in analysis

find_center(m32_img,suggested_m32_kernel,show_center=True)

# Now deal with the elliptical radii

"""The following function relies on q, ratio of minor to major axes and theta, position angle.
    M31: q=0.225, theta=52
    M32: q=0.8, theta=120
    M110: q=0.59 theta=100
    These are each approximations from the online resources I could find. 
    Since these galaxies are so far away, the location of the telescope on Earth shouldn't make a difference?
    """

m31_q=0.225
m31_theta=52
m32_q=0.8
m32_theta=120
m110_q=0.59
m110_theta=100

def elliptical_radius(image,center,q,theta):
    x0,y0=center
    y,x=np.indices(image.shape[:2]) #creating image arrays

    dx=x-x0 #getting coordinates with respect to center
    dy=y-y0
    theta=np.radians(theta) #converting to radians
    x_rot=dx*np.cos(theta)+dy*np.sin(theta) #rotating coordinates
    y_rot=-dx*np.sin(theta)+dy*np.cos(theta) #this rotation is basically us just

    radius=np.sqrt(x_rot**2+(y_rot/q)**2) #calculating radius
    return radius # we return an array containing the radius from center at every point

print(elliptical_radius(m31_gray,find_center(m31_img,suggested_m31_kernel,show_center=True),m31_q,m31_theta))

# now actually assign each radius a brightness. For every unique r, make a array or something of all the
# coordinates with that radius. Then, for each array, calculate the brightness (or grayscale intensity) in
# the main image. After doing that, average (?) the intensities across a given radius and then plot that.
# In theory, this should give us an exponential curve that we then just compare w/ the sersic profile.
