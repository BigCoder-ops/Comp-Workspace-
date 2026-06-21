import numpy as np
import scipy.linalg as la

# 1. Construct the diagonals for the tridiagonal matrix A
# The main diagonal is filled with (1 + 2r)
main_diag = (1 + 2*r) * np.ones(N-1)

# The upper and lower diagonals are filled with -r
off_diag = -r * np.ones(N-2)

# 2. Build the matrix A
A = np.diag(main_diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)

# 3. LU Decomposition
# Question 3 specifically requires solving this using LU decomposition.
# Since Matrix A is constant throughout time, it is highly efficient to
# compute the LU decomposition ONCE outside the time-stepping loop.
P, L_mat, U_mat = la.lu(A)