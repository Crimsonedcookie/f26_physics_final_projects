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
from scipy.optimize import curve_fit

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

find_center(m32_img,suggested_m32_kernel,show_center=False)

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

print(elliptical_radius(m31_gray,find_center(m31_img,suggested_m31_kernel,show_center=False),m31_q,m31_theta))

# now actually assign each radius a brightness. For every unique r, make a array or something of all the
# coordinates with that radius. Then, for each array, calculate the brightness (or grayscale intensity) in
# the main image. After doing that, average (?) the intensities across a given radius and then plot that.
# In theory, this should give us an exponential curve that we then just compare w/ the sersic profile.

# Way easier method actually: just use histograms on numpy to sort everything. Then average intensities within

def radial_profile(image,radius,bin_width=1):
    r=radius.ravel() #collapsing radius into 1d array
    intensity=image.astype(float).ravel() #collapsing intensities into 1d array

    bins=np.arange(0,r.max()+bin_width,bin_width) #making radial bins for histogram
    intensity_sum,bin_edges=np.histogram(r,bins=bins,weights=intensity) #calculate intensity in each bin
    pixel_count,_=np.histogram(r,bins=bins) #calculate # of pixels in each bin

    #avg_intensity=intensity_sum/pixel_count #avg intensity calculation
    avg_intensity=np.divide(intensity_sum,pixel_count,out=np.full_like(intensity_sum,np.nan,dtype=float),where=pixel_count>0)
    radius_centers=(bin_edges[:-1]+bin_edges[1:])/2
    return radius_centers, avg_intensity



m31_center=find_center(m31_img,suggested_m31_kernel,show_center=True)
m32_center=find_center(m32_img,suggested_m32_kernel,show_center=True)
m110_center=find_center(m110_img,suggested_m110_kernel,show_center=True)

m31_radius=elliptical_radius(m31_gray,m31_center,m31_q,m31_theta)
m32_radius=elliptical_radius(m32_gray,m32_center,m32_q,m32_theta)
m110_radius=elliptical_radius(m110_gray,m110_center,m110_q,m110_theta)

m31_r,m31_I=radial_profile(m31_gray, m31_radius)
m32_r,m32_I=radial_profile(m32_gray, m32_radius)
m110_r,m110_I=radial_profile(m110_gray, m110_radius)


# Now we just plot these guys!
def single_plot():
    plt.figure(figsize=(10,6))

    plt.plot(m31_r,m31_I,label="M31")
    plt.plot(m32_r,m32_I,label="M32")
    plt.plot(m110_r,m110_I,label="M110")

    plt.xlabel("Elliptical radius in pixels")
    plt.ylabel("Avg light intensity")
    plt.title("Galactic Radial Brightness Profile")
    plt.legend()
    plt.grid()



def separate_plots():
    fig,axes=plt.subplots(1,3,figsize=(15,5))
    data=[(m31_r,m31_I,"M31"),(m32_r,m32_I,"M32"),(m110_r,m110_I,"M110")]
    for ax, (r, I, name) in zip(axes, data):
        ax.plot(r, I)
        ax.set_title(name)
        ax.set_xlabel("Elliptical radius in pixels")
        ax.set_ylabel("Avg light intensity")
        ax.grid()
    plt.tight_layout()



separate_plots()


# Ok now the fun part is fitting this to Sersic Model
# need to define sersic model first

def sersic_profile(r,Ie,re,n,background):
    bn=2*n-1/3+4/(405*n)+46/(25515*n**2)
    return background+Ie*np.exp(-bn*((r/re)**(1/n)-1)) #background is a constant, the brightness that's not from the galaxy


# now make the fitting function

def fit_to_sersic_profile(r,intensity):
    valid=np.isfinite(r) & np.isfinite(intensity) #remove infinities and NaNs
    r=r[valid]
    intensity=intensity[valid]

    background_estimate=np.median(intensity[int(0.9*len(intensity)):]) #estimating that the background intensity from outer tenth of profile

    Ie_estimate=(np.max(intensity)-background_estimate)/2
    re_estimate=r[np.argmin(np.abs(intensity-(background_estimate+Ie_estimate)))]
    n_estimate=2

    initial_estimate=[Ie_estimate,re_estimate,n_estimate,background_estimate]

    #Fit sersic model
    parameters,covariance=curve_fit(sersic_profile,r,intensity,p0=initial_estimate,bounds=([0,0.1,0.3,0],[np.inf,np.inf,8,255]),maxfev=20000)

    return parameters,covariance

m31_params,m31_cov=fit_to_sersic_profile(m31_r,m31_I)
m32_params,m32_cov=fit_to_sersic_profile(m32_r,m32_I)
m110_params,m110_cov=fit_to_sersic_profile(m110_r,m110_I)

print("M31:",m31_params)
print("M32:",m32_params)
print("M110:",m110_params)


# now plot these against the other data we already found

def single_plot_w_sersic():
    plt.figure(figsize=(10,6))
    plt.scatter(m31_r,m31_I,s=3,label="M31 data") # plotting experimental data
    plt.scatter(m32_r,m32_I,s=3,label="M32 data")
    plt.scatter(m110_r,m110_I,s=3,label="M110 data")

    plt.plot(m31_r,sersic_profile(m31_r,*m31_params),label="M31 fit") # plotting expected data
    plt.plot(m32_r,sersic_profile(m32_r,*m31_params),label="M32 fit")
    plt.plot(m110_r,sersic_profile(m110_r,*m31_params),label="M110 fit")

    plt.xlabel("Elliptical radius in pixels")
    plt.ylabel("Avg light intensity")
    plt.title("Sersic Profile Fits")
    plt.legend()
    plt.grid()



def separate_plots_w_sersic():
    fig,axes=plt.subplots(1,3,figsize=(15,5))
    data=[(m31_r,m31_I,sersic_profile(m31_r,*m31_params),"M31"),(m32_r,m32_I,sersic_profile(m32_r,*m32_params),"M32"),(m110_r,m110_I,sersic_profile(m110_r,*m110_params),"M110")]
    for ax, (r, I, sp, name) in zip(axes, data):
        ax.scatter(r, I, color="blue", label="Experimental Data")
        ax.plot(r, sp, color="red", label="Sersic Model")
        ax.set_title(name)
        ax.set_xlabel("Elliptical radius in pixels")
        ax.set_ylabel("Avg light intensity")
        ax.grid()
        ax.legend()
    plt.tight_layout()


separate_plots_w_sersic()


# RMSE Error calculation time!

def calc_rmse(r,intensity,parameters):
    valid=np.isfinite(r) & np.isfinite(intensity)
    r=r[valid]
    intensity=intensity[valid]
    predicted=sersic_profile(r,*parameters)
    residuals=intensity-predicted
    rmse=np.sqrt(np.mean(residuals**2))
    return rmse

def calc_r_squared(r, intensity, parameters):
    valid=np.isfinite(r) & np.isfinite(intensity)
    r=r[valid]
    intensity=intensity[valid]
    predicted=sersic_profile(r,*parameters)
    ss_res=np.sum((intensity-predicted)**2)
    ss_tot=np.sum((intensity-np.mean(intensity))**2)
    return 1 - ss_res / ss_tot


m31_rmse=calc_rmse(m31_r,m31_I,m31_params)
m32_rmse=calc_rmse(m32_r,m32_I,m32_params)
m110_rmse=calc_rmse(m110_r,m110_I,m110_params)

m31_r_error=calc_r_squared(m31_r,m31_I,m31_params)
m32_r_error=calc_r_squared(m32_r,m32_I,m32_params)
m110_r_error=calc_r_squared(m110_r,m110_I,m110_params)

print("M31 error", m31_rmse, m31_r_error)
print("M32 error", m32_rmse, m32_r_error)
print("M110 error", m110_rmse, m110_r_error)

plt.show()