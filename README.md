# Instructions
## List of files :
### sparse_stoch_mat_central.py
Only contains one function : make_transition_matrix.
\
Given a set of parameters (see functions documentation) returns a stochastic matrix that account for all the transition in the system. We note the state of the system as :
$$\bar{\rho} = \{\rho_a(0), \rho_a(\text{d}x), ..., \rho_a(i\text{d}x), ..., \rho_a(L), \rho_b(0),...,\rho_b(L),...,\rho_c(0),...\rho_c(L)\}$$
If we note the matrix $\Omega$, then $\Omega_{i,j}$ is the transition of a particle from the site $i$ to $j$. It includes both the diffusion transition ($i\rightarrow i \pm 1$) and the chemical ones. We use fixed boundary conditions. The time evolution of the system is given by :
$$\partial_t\bar{\rho} = \Omega \bar{\rho}$$
Since $\Omega$ is a stochastic matrix, it is guaranteed to have a unique 0 eigen value which is the equilibrium density profile. We use the modified LU method to find a non-zero vector in the kernel. The equilibrium solution is parallel to this vector. The normalization condition fixes the overall amplitude of the vector.
\
Remarque: The matrix returned is a sparse matrix object.

### function.py

See documentation of the various functions in it.

### General_case_ooe_fluxes

Example of how to use this modules
