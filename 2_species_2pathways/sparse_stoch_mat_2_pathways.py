import numpy as np
from scipy.sparse import csr_matrix
# Define the M matrix :
def make_transition_matrix(Va,Vb, # potential
                            kab,kba, # the transition functions 
                            mu, # activity that drives the ooe
                            X0,Xf,n): # the spacial parameter of the problem
    """
    Va : are the potential that applies to the a,b and c particles
    dVa,dVb,dVc : are the respective derivate of the potential
    X0 : left most fixed boundary
    Xf : right most fixed boundary
    n : number of points used to discretize the space.
    
    returns : a matrix of size 3n * 3n that has to be a stochastic matrix.
    check -> np.sum(matrix, axis=0) = [0 for _ in range(3*n)]
    """
    #matrix = np.zeros((3*n,3*n),dtype=float)
    
    row_col_data = list()

    dx = (Xf-X0)/n
    for i in range(n):
        # make the "a" transition
        x = i*dx+X0
        dVxdx = (Va(x+2*dx)-Va(x))/(2*dx)
        dVxmdx = (Va(x)-Va(x-2*dx))/(2*dx)
        if i!=0 and i !=n-1:
            row_col_data.append([i,i,-2/(dx**2) - kab(x,mu)])
            row_col_data.append([i,n+i,kba(x)])
            row_col_data.append([i,i+1,dVxdx/(2*dx)+1/(dx)**2])
            row_col_data.append([i,i-1,-dVxmdx/(2*dx)+1/(dx)**2])


        elif i==0:            
            row_col_data.append([i,i,-1/(dx**2)  - kab(x,mu)])
            row_col_data.append([i,n+i,kba(x)])
            row_col_data.append([i,i+1,dVxdx/(2*dx)+1/(dx)**2])

        elif i==n-1:
            row_col_data.append([i,i,-1/(dx**2)  - kab(x,mu)])
            row_col_data.append([i,n+i,kba(x)])
            row_col_data.append([i,i-1,-dVxmdx/(2*dx)+1/(dx)**2])

    for i in range(n,2*n):
        # make the "b" transition
        x = (i-n)*dx+X0
        dVxdx = (Vb(x+2*dx)-Vb(x))/(2*dx)
        dVxmdx = (Vb(x)-Vb(x-2*dx))/(2*dx)
        if i!=n and i!=2*n-1:

            row_col_data.append([i,i,-2/(dx**2) - kba(x)])
            row_col_data.append([i,i-n,kab(x,mu)])
            row_col_data.append([i,i+1,dVxdx/(2*dx)+1/(dx)**2])
            row_col_data.append([i,i-1,-dVxmdx/(2*dx)+1/(dx)**2])
        elif i==n:
            row_col_data.append([i,i,-1/(dx**2) - kba(x)])
            row_col_data.append([i,i-n,kab(x,mu)])
            row_col_data.append([i,i+1,dVxdx/(2*dx)+1/(dx)**2])

        elif i==2*n-1:
            row_col_data.append([i,i,-1/(dx**2) - kba(x)])
            row_col_data.append([i,i-n,kab(x,mu)])
            row_col_data.append([i,i-1,-dVxmdx/(2*dx)+1/(dx)**2])

    row_col_data = np.array(row_col_data)
    matrix = csr_matrix((row_col_data[:,2],(row_col_data[:,0],row_col_data[:,1])),shape=(2*n,2*n),dtype=float)

    return matrix