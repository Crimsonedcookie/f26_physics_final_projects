import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

"""step 1: define some variables"""

r=50 # radius of pipe for time being
n=101 # number of elements in each axis
delta_p=5 # pressure drop
length=500 # length of section of pipe
mu=0.001 # viscosity

"""step 2: create the cross-sectional grid and break into discrete blocks from which to find the velocity"""

# create a 2d array, first element is an array of x coords and second is an array of y coords
y=np.linspace(-r,r,n)
z=np.linspace(-r,r,n) #🫠
y,z=np.meshgrid(y,z)

# using pythag theorem + masks to select only the blocks whose positions are within the radius
mask=y**2+z**2<=r**2

# assign an index to each interior grid points
index=np.full((n,n),-1) # -1 will represent all the points that are outside the pipe
index[mask]=np.arange(np.sum(mask)) # assign integer value to every interior point

num_interior=np.sum(mask)
print("num_interior",num_interior)

"""step 3: create velocity field"""

# assign each grid space a velocity (for right now, zero)
u=np.zeros((n,n))
u[~mask]=np.nan # replacing any velocities for points outside the pipe with nan

# we will be updating this velocity field as we calculate with the poisson velocity equation

"""step 4: perform calculations (using finite difference method)"""

# Derived the finite difference method approximated version of the equation (i.e. simplifying the partial derivatives)
# Now, we have an equation that describes the velocity at a given point as dependent on the four adjacent points
# Going to use scipy's sparse matrix solver to deal with this equation

# start by making our coefficient matrix
A=lil_matrix((num_interior,num_interior))

# this is a complicated step; basically matrix A has num_interior rows of num_interior length; each row has a single 4 and four -1s (to represent
# the specific point we're looking at. Everything else is zero. Since we have num_interior rows, we are essentially creating this 1D array
# with every point eventually being the one we're focusing on.
for i in range(n):
    for j in range(n):
        if mask[i,j]:
            p=index[i,j]
            A[p,p]=4 # setting coefficient of selected point to 4, as we derived
            adj_points=[(i-1,j),(i+1,j),(i,j-1),(i,j+1)]
            for ni, nj in adj_points:
                if 0<=ni<n and 0<=nj<n: # checking if neighbors are even interior points (or if they're on the boundary)
                    if mask[ni,nj]:
                        q=index[ni,nj]
                        A[p,q]=-1 # setting coefficient of adj point to -1, as derived
A=A.tocsr()

# ok now we create the b vector
G=delta_p/length # calculating pressure gradient
h=((2*r)+1)/n # setting step length
B=np.full(num_interior,(G*h**2)/mu)
print(B)

# now, we use the sparse matrix solver from scipy
u_vector=spsolve(A,B) # this is a 1D array
u[mask]=u_vector # now we applied it onto the 2D array we had


#visualize countour field
plt.contourf(y,z,u,levels=50)
plt.colorbar(label="Velocity")
plt.xlabel("y")
plt.ylabel("z")
plt.axis("equal")
plt.show()
