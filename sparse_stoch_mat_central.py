from xml.dom import IndexSizeErr
import numpy as np
from scipy.sparse import csr_matrix
# Define the M matrix :
def make_transition_matrix(Va,Vb,Vc, # potential
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
    #matrix = np.zeros((3*n,3*n),dtype=float)
    
    row = list()#np.zeros(((n-2)*5+2*4)*3,dtype=int)
    col = list()
    data = list()
    row_col_data = list()

    dx = (Xf-X0)/n
    for i in range(n):
        # make the "a" transition
        x = i*dx+X0
        dVxdx = (Va(x+2*dx)-Va(x))/(2*dx)
        dVxmdx = (Va(x)-Va(x-2*dx))/(2*dx)
        if i!=0 and i !=n-1:
            #matrix[i,i] += -2/(dx**2) - kac(x,Aac) - kab(x,Aab)
            #matrix[i,n+i] += kba(x)
            #matrix[i,2*n+i] += kca(x)
            #matrix[i,i+1] += dVxdx/(2*dx)
            #matrix[i,i-1] += -dVxmdx/(2*dx)
            #matrix[i,i+1] += 1/(dx)**2
            #matrix[i,i-1] += 1/(dx)**2

            row_col_data.append([i,i,-2/(dx**2) - kac(x,Aac) - kab(x,Aab)])
            row_col_data.append([i,n+i,kba(x)])
            row_col_data.append([i,2*n+i,kca(x)])
            row_col_data.append([i,i+1,dVxdx/(2*dx)+1/(dx)**2])
            row_col_data.append([i,i-1,-dVxmdx/(2*dx)+1/(dx)**2])


        elif i==0:
            #matrix[i,i] += -1/(dx**2) - kac(x,Aac) - kab(x,Aab)
            #matrix[i,n+i] += kba(x)
            #matrix[i,2*n+i] += kca(x)
            #matrix[i,i+1] += dVxdx/(2*dx)
            #matrix[i,i+1] += 1/(dx)**2

            row_col_data.append([i,i,-1/(dx**2) - kac(x,Aac) - kab(x,Aab)])
            row_col_data.append([i,n+i,kba(x)])
            row_col_data.append([i,2*n+i,kca(x)])
            row_col_data.append([i,i+1,dVxdx/(2*dx)+1/(dx)**2])
            #row_col_data([i,i-1,-dVxmdx/(2*dx)+1/(dx)**2])

        elif i==n-1:
            #matrix[i,i] += -1/(dx**2) - kac(x,Aac) - kab(x,Aab)
            #matrix[i,n+i] += kba(x)
            #matrix[i,2*n+i] += kca(x)
            #matrix[i,i-1] += -dVxmdx/(2*dx)
            #matrix[i,i-1] += 1/(dx)**2

            row_col_data.append([i,i,-1/(dx**2) - kac(x,Aac) - kab(x,Aab)])
            row_col_data.append([i,n+i,kba(x)])
            row_col_data.append([i,2*n+i,kca(x)])
            #row_col_data.append([i,i+1,dVxdx/(2*dx)+1/(dx)**2])
            row_col_data.append([i,i-1,-dVxmdx/(2*dx)+1/(dx)**2])

    """ try:
            if i+1>n-1:
                raise IndexError
            matrix[i,i+1] += dVxdx/(2*dx)
        except IndexError:
            pass
        try:
            matrix[i,i-1] += -dVxmdx/(2*dx)
        except IndexError:
            pass
        try:
            if i+2>n-1:
                matrix[i,i+2] += 1/(2*dx)**2
        except IndexError:
            pass
        try:
            matrix[i,i-2] += 1/(2*dx)**2
        except IndexError:
            pass """
    for i in range(n,2*n):
        # make the "b" transition
        x = (i-n)*dx+X0
        dVxdx = (Vb(x+2*dx)-Vb(x))/(2*dx)
        dVxmdx = (Vb(x)-Vb(x-2*dx))/(2*dx)
        if i!=n and i!=2*n-1:
            #matrix[i,i] += -2/(dx**2) - kba(x) - kbc(x,Abc)
            #matrix[i,i-n] += kab(x,Aab)
            #matrix[i,i+n] += kcb(x)
            #matrix[i,i+1] += dVxdx/(2*dx)
            #matrix[i,i-1] += -dVxmdx/(2*dx)
            #matrix[i,i+1] += 1/(dx)**2
            #matrix[i,i-1] += 1/(dx)**2

            row_col_data.append([i,i,-2/(dx**2) - kba(x) - kbc(x,Abc)])
            row_col_data.append([i,i-n,kab(x,Aab)])
            row_col_data.append([i,i+n,kcb(x)])
            row_col_data.append([i,i+1,dVxdx/(2*dx)+1/(dx)**2])
            row_col_data.append([i,i-1,-dVxmdx/(2*dx)+1/(dx)**2])
        elif i==n:
            #matrix[i,i] += -1/(dx**2) - kba(x) - kbc(x,Abc)
            #matrix[i,i-n] += kab(x,Aab)
            #matrix[i,i+n] += kcb(x)
            #matrix[i,i+1] += dVxdx/(2*dx)
            #matrix[i,i+1] += 1/(dx)**2

            row_col_data.append([i,i,-1/(dx**2) - kba(x) - kbc(x,Abc)])
            row_col_data.append([i,i-n,kab(x,Aab)])
            row_col_data.append([i,i+n,kcb(x)])
            row_col_data.append([i,i+1,dVxdx/(2*dx)+1/(dx)**2])
            #row_col_data([i,i-1,-dVxmdx/(2*dx)+1/(dx)**2])

        elif i==2*n-1:
            #matrix[i,i] += -1/(dx**2) - kba(x) - kbc(x,Abc)
            #matrix[i,i-n] += kab(x,Aab)
            #matrix[i,i+n] += kcb(x)
            #matrix[i,i-1] += -dVxmdx/(2*dx)
            #matrix[i,i-1] += 1/(dx)**2

            row_col_data.append([i,i,-1/(dx**2) - kba(x) - kbc(x,Abc)])
            row_col_data.append([i,i-n,kab(x,Aab)])
            row_col_data.append([i,i+n,kcb(x)])
            #row_col_data.append([i,i+1,dVxdx/(2*dx)+1/(dx)**2])
            row_col_data.append([i,i-1,-dVxmdx/(2*dx)+1/(dx)**2])

    for i in range(2*n,3*n):
        # make the "c" transition
        x = (i-2*n)*dx+X0
        dVxdx = (Vc(x+2*dx)-Vc(x))/(2*dx)
        dVxmdx = (Vc(x)-Vc(x-2*dx))/(2*dx)
        if i!=2*n and i != 3*n -1:
            #matrix[i,i] += -2/(dx**2) - kca(x) - kcb(x)
            #matrix[i,i-2*n] += kac(x,Aac)
            #matrix[i,i-n] += kbc(x,Abc)
            #matrix[i,i+1] += dVxdx/(2*dx)
            #matrix[i,i-1] += -dVxmdx/(2*dx)
            #matrix[i,i+1] += 1/(dx)**2
            #matrix[i,i-1] += 1/(dx)**2

            row_col_data.append([i,i,-2/(dx**2) - kca(x) - kcb(x)])
            row_col_data.append([i,i-2*n,kac(x,Aac)])
            row_col_data.append([i,i-n,kbc(x,Abc)])
            row_col_data.append([i,i+1,dVxdx/(2*dx)+1/(dx)**2])
            row_col_data.append([i,i-1,-dVxmdx/(2*dx)+1/(dx)**2])
        elif i==2*n:
            #matrix[i,i] += -1/(dx**2) - kca(x) - kcb(x)
            #matrix[i,i-2*n] += kac(x,Aac)
            #matrix[i,i-n] += kbc(x,Abc)
            #matrix[i,i+1] += dVxdx/(2*dx)
            #matrix[i,i+1] += 1/(dx)**2

            row_col_data.append([i,i,-1/(dx**2) - kca(x) - kcb(x)])
            row_col_data.append([i,i-2*n,kac(x,Aac)])
            row_col_data.append([i,i-n,kbc(x,Abc)])
            row_col_data.append([i,i+1,dVxdx/(2*dx)+1/(dx)**2])
            #row_col_data([i,i-1,-dVxmdx/(2*dx)+1/(dx)**2])
        elif i==3*n-1:
            #matrix[i,i] += -1/(dx**2) - kca(x) - kcb(x)
            #matrix[i,i-2*n] += kac(x,Aac)
            #matrix[i,i-n] += kbc(x,Abc)
            #matrix[i,i-1] += -dVxmdx/(2*dx)
            #matrix[i,i-1] += 1/(dx)**2
            row_col_data.append([i,i,-1/(dx**2) - kca(x) - kcb(x)])
            row_col_data.append([i,i-2*n,kac(x,Aac)])
            row_col_data.append([i,i-n,kbc(x,Abc)])
            #row_col_data.append([i,i+1,dVxdx/(2*dx)+1/(dx)**2])
            row_col_data.append([i,i-1,-dVxmdx/(2*dx)+1/(dx)**2])

    row_col_data = np.array(row_col_data)
    matrix = csr_matrix((row_col_data[:,2],(row_col_data[:,0],row_col_data[:,1])),shape=(3*n,3*n),dtype=float)

    return matrix