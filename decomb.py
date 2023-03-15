import numpy as np
from scipy.linalg import svdvals, diagsvd
import scipy.linalg as la

# Define a 4x4 matrix A
A = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]])
u,s,v = np.linalg.svd(A)
print(u@la.diagsvd(s,*A.shape)@v)

# Reshape A into a 2x2 matrix B whose entries are also matrices
B = A.reshape(2,-1)
# Transpose B so that it becomes a stack of matrices
B = B.T
# Compute only the singular values of B
S = svdvals(B) 
# Construct U and V from S using diagsvd
U = diagsvd(S[:2], 4, 2)
V = diagsvd(S[:2], 4, 2)

# Extract the singular values and normalize them
s = S / S.sum()

# Reshape U and V into matrices whose columns are operators on each subsystem
U = U.reshape(4,-1)
V = V.reshape(4,-1)

print(f"A{A.shape}:\n{A}")
print(f"s{s.shape}:\n{s}")
print(f"U{U.shape}:\n{U}")
print(f"V{V.shape}:\n{V}")
# Display the operator Schmidt decomposition of A
print('The operator Schmidt decomposition of A is:')
for i in range(2):
    print(f'{str(s[i])} * ({str(U[:,i])}) * ({str(V[:,i])}) = {s[i]*U[:,i]*V[:,i]}')
# Reconstruct the 4x4 matrix
A_reconstructed = np.matmul(U, np.matmul(np.diag(s), V.T))
print(f"\nA_reconstructed{A_reconstructed.shape}:\n{A_reconstructed}")
