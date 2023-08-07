import numpy as np
import sparse_stoch_mat_2_pathways as sp_stoch_mat
from functions import *

class OneD_Phase_sep:
    def __init__(self,Vs,Es,steeps,mu,size,X0,Xf,n) -> None:
        self.VA0,self.VB0,self.VAB01,self.VAB02= Vs[0],Vs[1],Vs[2],Vs[3]
        self.EAB01,self.EAB02 = Es[0],Es[1]
        self.steepA,self.steepB,self.steepAB1,self.steepAB2 = steeps[0],steeps[1],steeps[2],steeps[3]
        self.mu = mu
        self.kab0=1.
        self.X0,self.Xf,self.n = X0,Xf,n
        self.dx = (Xf-X0)/n
        self.X = np.linspace(X0,Xf,n,endpoint=False)

        self.va = lambda x : self.VA0/2*(1+np.tanh((x-size/2)*self.steepA))
        self.vb = lambda x : self.VB0/2*(1+np.tanh((x-size/2)*self.steepB))
        self.vab1 = lambda x : self.VAB01/2*(1+np.tanh((x-size/2)*self.steepAB1))+self.EAB01
        self.vab2 = lambda x : self.VAB02/2*(1+np.tanh((x-size/2)*self.steepAB2))+self.EAB02

        self.kab1 = lambda X,*arg : np.array([self.kab0*(np.exp(-(self.vab1(x)-self.va(x)))) for x in X]) if type(X)==np.ndarray else self.kab0*(np.exp(-(self.vab1(X)-self.va(X))))
        self.kab2 = lambda X,*arg : np.array([self.kab0*(np.exp(-(self.vab2(x)-self.va(x))+arg[0])) for x in X]) if type(X)==np.ndarray else self.kab0*(np.exp(-(self.vab2(X)-self.va(X))+arg[0]))

        self.kba1 = lambda X,*arg : np.array([self.kab0*(np.exp(-(self.vab1(x)-self.vb(x)))) for x in X]) if type(X)==np.ndarray else self.kab0*(np.exp(-(self.vab1(X)-self.vb(X))))
        self.kba2 = lambda X,*arg : np.array([self.kab0*(np.exp(-(self.vab2(x)-self.vb(x)))) for x in X]) if type(X)==np.ndarray else self.kab0*(np.exp(-(self.vab2(X)-self.vb(X))))

        self.kab = lambda X,*arg : self.kab1(X,*arg)+self.kab2(X,*arg)
        self.kba = lambda X,*arg : self.kba1(X,*arg)+self.kba2(X,*arg)

    def get_analytical_lambda(self,left=False,right=False):
        if left:
            x = self.X[0]
        elif right:
            x = self.X[-1]
        else:
            raise ValueError("left or right should be true")
        return 1/np.sqrt(self.kab1(x,self.mu)+self.kab2(x,self.mu)+self.kba1(x,self.mu)+self.kba2(x,self.mu))
    def get_lambda(self,left=False,right=False):
        if left:
            x = self.X[0]
        elif right:
            x = self.X[-1]
        else:
            raise ValueError("left or right should be true")
        matrix = np.array([
            [-self.kab1(x,self.mu) - self.kab2(x,self.mu), self.kba1(x,self.mu) + self.kba2(x,self.mu)],
            [-self.kba1(x,self.mu) - self.kba2(x,self.mu), self.kab1(x,self.mu) + self.kab2(x,self.mu)]
            ])
        # Try to diagonalize the matrix
        try:
            eigenvalues, eigenvectors = np.linalg.eig(matrix)
            return eigenvalues.real
        except np.linalg.LinAlgError:
            # Matrix is not diagonalizable
            return None
    def compute_rhos(self):
        Ooe_Stoch_Mat = sp_stoch_mat.make_transition_matrix(self.va,self.vb,self.kab,self.kba,self.mu,self.X0,self.Xf,self.n)
        self.rhoa,self.rhob = get_kernel_stoch_mat(Ooe_Stoch_Mat,self.dx)
    def get_diff_fluxs(self):
        if not hasattr(self,'rhoa'):
            self.compute_rhos()
        return diff_flux(self.rhoa,self.va(self.X),self.dx),diff_flux(self.rhob,self.vb(self.X),self.dx)
    def get_chem_fluxs(self):
        IX = np.array([x//self.dx for x in self.X],dtype=int)
        #return np.array([self.rhoa[ix]*self.kab1(ix*self.dx,self.mu)-self.rhob[ix]*self.kba2(ix*self.dx,self.mu) for ix in IX]),np.array([self.rhoa[ix]*self.kab2(ix*self.dx,self.mu)-self.rhob[ix]*self.kba1(ix*self.dx,self.mu) for ix in IX])
        return np.array([self.rhoa[ix]*(self.kab1(ix*self.dx,self.mu)-self.kab2(ix*self.dx,self.mu))+self.rhob[ix]*(-self.kba1(ix*self.dx,self.mu)+self.kba2(ix*self.dx,self.mu)) for ix in IX])
    def get_prod(self):
        IX = np.array([x//self.dx for x in self.X],dtype=int)
        return np.array([self.rhoa[ix]*(self.kab1(ix*self.dx,self.mu)+self.kab2(ix*self.dx,self.mu))-self.rhob[ix]*(self.kba2(ix*self.dx,self.mu)+self.kba1(ix*self.dx,self.mu))for ix in IX])
    