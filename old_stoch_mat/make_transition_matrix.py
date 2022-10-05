import numpy as np
# Define the M matrix :
def make_transition_matrix(dVa,dVb,dVc, # potential
                            kab,kba,kbc,kcb,kac,kca, # the transition functions 
                            kab0,kbc0,kac0, # the transition rates
                            Aab,Abc,Aac, # activity that drives the ooe
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
    matrix = np.zeros((3*n,3*n),dtype=float)
    dx = (Xf-X0)/n
    for i in range(n):
        # make the "a" transition
        x = i*dx+X0
        if i!=0 and i!=n-1:
            matrix[i,i] += -2/dx**2 -dVa(x)/dx - kac(x,Aac) - kab(x,Aab)
            matrix[i,n+i] += kba(x)
            matrix[i,2*n+i] += kca(x)
            matrix[i,i+1] += 1/dx**2+dVa(x+dx)/dx
            matrix[i,i-1] += 1/dx**2
        elif i == 0:
            # boundary condition in x = x0 and Xf
            matrix[i,i] += -1/dx**2 - kac(x,Aac) - kab(x,Aab)
            matrix[i,n+i] += kba(x)
            matrix[i,2*n+i] += kca(x)
            matrix[i,i+1] += 1/dx**2+dVa(x+dx)/dx
        elif i == n-1:
            # boundary condition in x = x0 and Xf
            matrix[i,i] += -1/dx**2 -dVa(x)/dx - kac(x,Aac) - kab(x,Aab)
            matrix[i,n+i] += kba(x)
            matrix[i,2*n+i] += kca(x)
            matrix[i,i-1] += 1/dx**2
    for i in range(n,2*n):
        # make the "b" transition
        x = (i-n)*dx+X0
        if i!=n and i!=2*n-1:
            matrix[i,i] += -2/dx**2 -dVb(x)/dx - kba(x) - kbc(x,Abc)
            matrix[i,i-n] += kab(x,Aab)
            matrix[i,i+n] += kcb(x)
            matrix[i,i+1] += 1/dx**2+dVb(x+dx)/dx
            matrix[i,i-1] += 1/dx**2
        elif i==n :
            matrix[i,i] += -1/dx**2 - kba(x) - kbc(x,Abc)
            matrix[i,i-n] += kab(x,Aab)
            matrix[i,i+n] += kcb(x)
            matrix[i,i+1] += 1/dx**2+dVb(x+dx)/dx

        elif i==2*n-1:
            matrix[i,i] += -1/dx**2 -dVb(x)/dx - kba(x) - kbc(x,Abc)
            matrix[i,i-n] += kab(x,Aab)
            matrix[i,i+n] += kcb(x)
            matrix[i,i-1] += 1/dx**2
    for i in range(2*n,3*n):
        # make the "c" transition
        x = (i-2*n)*dx+X0
        if i!=2*n and i!=3*n-1:
            matrix[i,i] += -2/dx**2 -dVc(x)/dx - kca(x) - kcb(x)
            matrix[i,i-2*n] += kac(x,Aac)
            matrix[i,i-n] += kbc(x,Abc)
            matrix[i,i+1] += 1/dx**2+dVc(x+dx)/dx
            matrix[i,i-1] += 1/dx**2
        elif i==2*n:
            matrix[i,i] += -1/dx**2 - kca(x) - kcb(x)
            matrix[i,i-2*n] += kac(x,Aac)
            matrix[i,i-n] += kbc(x,Abc)
            matrix[i,i+1] += 1/dx**2+dVc(x+dx)/dx
        elif i==3*n-1:
            matrix[i,i] += -1/dx**2 -dVc(x)/dx - kca(x) - kcb(x)
            matrix[i,i-2*n] += kac(x,Aac)
            matrix[i,i-n] += kbc(x,Abc)
            matrix[i,i-1] += 1/dx**2
    return matrix