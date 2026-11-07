import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg as spsolve

"""step 1: define some variables"""

r=50 # radius of pipe for time being
n=101 # number of elements in each axis

"""step 2: create the cross-sectional grid and break into discrete blocks from which to find the velocity"""

# create a 2d array, first element is an array of x coords and second is an array of y coords
y=np.linspace(-r,r,n)
z=np.linspace(-r,r,n) #🫠
grid=np.meshgrid(y,z)

# using pythag theorem + masks to select only the blocks whose positions are within the radius
mask=y**2+z**2<=r**2
y_masked=np.where(mask,y,np.nan)
z_masked=np.where(mask,z,np.nan)
inside_grid=np.meshgrid(y_masked,z_masked)

"""step 3: create velocity field"""

# assign each grid space a velocity (for right now, zero)
u=np.zeros(n,n)

# we will be updating this velocity field as we calculate with the poisson velocity equation

"""step 4: perform calculations (using finite difference method)"""

# Derived the finite difference method approximated version of the equation (i.e. simplifying the partial derivatives)
# Now, we have an equation that describes the velocity at a given point as dependent on the four adjacent points
# Going to use scipy's sparse matrix solver to deal with this equation

# start by making our coefficient matrix
