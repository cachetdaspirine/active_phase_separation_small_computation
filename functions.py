import numpy as np
from scipy.sparse.linalg import spsolve

def D(rho,dx):
    """
    return the derivative of the density along space
    """
    drho = np.zeros(rho.shape[0],dtype=float)
    for i in range(1,rho.shape[0]-1):
        drho[i] = (rho[i+1]-rho[i-1])/(2*dx)
    drho[0] = 0.
    drho[-1] = 0.
    return drho
def DD(rho,dx):
    """
    return the second derivative of the density along space
    """
    drho = np.zeros(rho.shape[0],dtype=float)
    for i in range(1,rho.shape[0]-1):
        drho[i] = (rho[i+1]+rho[i-1]-2*rho[i])/(dx)**2
    drho[0] = 0.
    drho[-1] = 0.
    return drho


def get_kernel_stoch_mat(Stoch_Mat,dx):
    """
    This function returns a non-zero vector from the kernel of the matrix.
    The matrix is assumed to be a stochastic matrix so that the kernel is of dimension 1.
    The matrix is assumed to represent three species, and returns three density profiles.
    the vector is normalized assuming that the sum of all species accross space is 1.
    """
    # use a reduced matrix (the first row and column is useless since the first value is fixed)
    Omega1 = Stoch_Mat[1:,1:]
    B = Stoch_Mat[1:,0]
    x = spsolve(Omega1,-B)

    # now add the first value as 1
    rho_lu = np.zeros(x.shape[0]+1,dtype=float)
    rho_lu[0] = 1
    rho_lu [1:] = x
    # normalize
    rho_lu = abs(rho_lu)/(sum(abs(rho_lu))*dx)

    n = Stoch_Mat.shape[0]//3

    rho_a = np.real(rho_lu[:n])
    rho_b = np.real(rho_lu[n:2*n])
    rho_c = np.real(rho_lu[2*n:])

    return rho_a,rho_b,rho_c

def make_eq_distrib(Va,Vb,Vc,x):
    """
    return the non-normalized distribution that is the solution of the diffusive equation
    """
    rho_a,rho_b,rho_c = np.exp(-Va(x)),np.exp(-Vb(x)),np.exp(-Vc(x))
    #Z = (sum(rho_a)+sum(rho_b)+sum(rho_c))*dx
    #res = np.zeros(3*n,dtype=float)
    return rho_a,rho_b,rho_c
def make_chem_eq(kab,kba,kac,kca,kbc,kcb,Aab,Aac,Abc,x):
    """
    return the non-normalized distribution that is the solution of the chemical equation
    """
    rho_a = kca(x,Abc)*kba(x)+kba(x)*kca(x)+kbc(x,Abc)*kca(x)
    rho_b = kab(x,Aab)*kcb(x)+kca(x)*kab(x,Aab)+kac(x,Aac)*kcb(x)
    rho_c = kbc(x,Abc)*kac(x,Aac)+kba(x)*kac(x,Aac)+kab(x,Aab)*kbc(x,Abc)
    #S = rho_a+rho_b+rho_c
    return rho_a,rho_b,rho_c

# define the two fluxes :
def chem_flux(kjis,kijs,rho_i,rho_js,ix,x):
    """
    - kjis : is a vector of chemical transition from any j to the specie i
    - kijs : is a vector of chemical transition from the specie i to any j
    - rho_i : is the array of density in space of the specie i
    - rho_js : is the array of array of density in space of all the j's species
    - x : is the position in space
    - ix : is the corresponding index
    !!!!! the kjis, kijs, and rho_js must be sorted similarly
    """
    if rho_js.shape[0] != kjis.shape[0] or rho_js.shape[0] != kjis.shape[0] :
        raise IndexError
    return sum([kjis[j](x)*rho_js[j][ix]-kijs[j](x)*rho_i[ix] for j in range(rho_js.__len__())])
def diff_flux(rho,V,dx):
    """"
    - rho : is the vector of density in space
    - V is a vector with the value of the potential in the space
    - rho[i] and V[i] are assumed to correspond to the same point in space
    - derivatives at the boundaries are assumed to be 0
    """
    dVx = D(V,dx)
    return D(rho,dx)+rho*dVx