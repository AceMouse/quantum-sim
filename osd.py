import numpy as np
import scipy.linalg as la
from scipy.linalg  import svdvals, diagsvd

import math
import cmath

_0     = [np.array([[1],[0]], dtype=complex), np.array([1,0], dtype=complex)]
_1     = [np.array([[0],[1]], dtype=complex), np.array([0,1], dtype=complex)]
_plus  = [(_0[i] + _1[i])/math.sqrt(2) for i in range(len(_0))]
_minus = [(_0[i] - _1[i])/math.sqrt(2) for i in range(len(_0))]

_I = np.array([[1,0],[0,1]], dtype=complex)
_X = np.array([[0,1],[1,0]], dtype=complex)
_H = (1/math.sqrt(2))*np.array([[1,1],[1,-1]], dtype=complex)
def kron(arr, dtype=complex):
    res = np.array([1],dtype=dtype)
    for x in arr:
        res = np.kron(res, x)
    return res

def _R(angle):
    return np.array([[1,0],[0,cmath.exp(1j*angle)]], dtype=complex)

def _R_n(n):
    return _R(2*math.pi/math.pow(2,n))

def I(k):
    return kron([_I]*k)
#return a matrix that applies gate U to the t'th qubit in an n qubit state.
def U(_U, t, n): 
    proj0 = _0[1]*_0[0]
    proj1 = _1[1]*_1[0]
    before = I(t-1) 
    after = I(n-t)    
    return kron([before,_U,after])
    
# https://quantumcomputing.stackexchange.com/a/4255 <- math
#return a matrix that applies gate U to the t'th qubit, controled by the c'th qubit, in an n qubit state 
def C(_U, c, t, n): 
    if c == t:
        raise Exception(f'Conditional Error: control and target are the same "{c} == {t}"')
    proj0 = _0[1]*_0[0]
    proj1 = _1[1]*_1[0]
    _min = min(c,t)
    _max = max(c,t)
    before = I(_min-1)    
    uninvolved = I(_max-_min-1)
    after = I(n-_max)
    x = [proj0, _I]
    y = [proj1, _U]
    if t < c:
        x = x[::-1]
        y = y[::-1]
    a = kron([before,x[0],uninvolved,x[1],after])
    b = kron([before,y[0],uninvolved,y[1],after])
    return a+b


def swapBits(x, p1, p2, n):
 
    # Move all bits of first
    # set to rightmost side
    set1 =  (x >> p1) & ((1<< n) - 1)
  
    # Move all bits of second
    # set to rightmost side
    set2 =  (x >> p2) & ((1 << n) - 1)
  
    # XOR the two sets
    xor = (set1 ^ set2)
  
    # Put the xor bits back
    # to their original positions
    xor = (xor << p1) | (xor << p2)
  
      # XOR the 'xor' with the
      # original number so that the
      # two sets are swapped
    result = x ^ xor
  
    return result

#CNOT after transformation
#  0, 1, 4, 5    0, 1, 2, 3
#  2, 3, 6, 7 => 4, 5, 6, 7
#  8, 9,12,13    8, 9,10,11
# 10,11,14,15   12,13,14,15
def transform(A):
    B=A.flatten()
    for i,x in enumerate(A.flatten()):
        B[swapBits(i,1,2,1)] = x
    B = B.reshape(-1,4)
    return B
    

A = C(_X,0,1,1)
print(f"A(ijkl):\n{A}\n")
A=transform(A)
print(f"A(ikjl):\n{A}\n")
u,s,vh = np.linalg.svd(A)
print(f"u:\n{u}\n")
print(f"vh:\n{vh}\n")
print(f"s:\n{s}\n")

ss=np.sqrt(s)
print(f"ss:\n{ss}\n")
ssd=la.diagsvd(ss,*A.shape)
print(f"ssd:\n{ssd}\n")

U=u@ssd
Vh=ssd@vh
print(f"u*ssd  = U :\n{U}\n")
print(f"ssd*vh = Vh:\n{Vh}\n")

print(f"U*Vh:\n{U@Vh}\n")
print(f"Transformed:\n{transform(U@Vh)}\n\n\n")
print(f"U reshaped to (i:2,k:2,j:4):\n{U.reshape(2,2,4)}\n")
print(f"U indecies swapped to (j:4,i:2,k:2):\n{U.reshape(2,2,4).transpose([2,0,1])}\n")
print(f"Vh reshaped to (j:4,l:2,m:2):\n{Vh.reshape(4,2,2)}\n")
import quimb as qu
import quimb.tensor as qtn
n=2
psi = qtn.tensor_builder.MPS_computational_state(input()) #all 0's state
print(f"psi:\n{psi.to_dense()}\n")
mpo = qtn.tensor_builder.MatrixProductOperator([U.reshape(2,2,4).transpose([2,0,1]),Vh.reshape(4,2,2)])

print(f"apply mpo:\n{mpo}\n")
psi = mpo.apply(psi)
print(f"psi:\n{psi.to_dense()}\n")

