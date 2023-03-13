import math
import cmath
import numpy as np

_0     = [np.array([[1],[0]], dtype=complex), np.array([1,0], dtype=complex)]
_1     = [np.array([[0],[1]], dtype=complex), np.array([0,1], dtype=complex)]
_plus  = [(_0[i] + _1[i])/math.sqrt(2) for i in range(len(_0))]
_minus = [(_0[i] - _1[i])/math.sqrt(2) for i in range(len(_0))]

_I = np.array([[1,0],[0,1]], dtype=complex)
_X = np.array([[0,1],[1,0]], dtype=complex)
_H = (1/math.sqrt(2))*np.array([[1,1],[1,-1]], dtype=complex)
def _R(angle):
    return np.array([[1,0],[0,cmath.exp(1j*angle)]], dtype=complex)

def _R_n(n):
    return _R(2*math.pi/math.pow(2,n))

def I(k):
    I_n = np.array([1],dtype=complex)
    for i in range(k):
        I_n = np.kron(I_n,_I)
    return I_n

#return a matrix that applies gate U to the t'th qubit in an n qubit state. (1 indexed)
def U(_U, t, n): 
    proj0 = _0[1]*_0[0]
    proj1 = _1[1]*_1[0]
    before = I(t-1) 
    after = I(n-t-1)    
    return np.kron(before,np.kron(_U,after))
    
# https://quantumcomputing.stackexchange.com/a/4255 <- math
#return a matrix that applies gate U to the t'th qubit, controled by the c'th qubit, in an n qubit state. (1 indexed) 
def C(_U, c, t, n): 
    proj0 = _0[1]*_0[0]
    proj1 = _1[1]*_1[0]
    _min = min(c,t)
    _max = max(c,t)
    before = I(_min-1)    
    uninvolved = I(_max-_min-1)
    after = I(n-_max-1)
    if c < t:
        a = np.kron(before,np.kron(proj0,np.kron(uninvolved,np.kron(_I,after))))
        b = np.kron(before,np.kron(proj1,np.kron(uninvolved,np.kron(_U,after))))
    elif t < c:
        a = np.kron(before,np.kron(_I,np.kron(uninvolved,np.kron(proj0,after))))
        b = np.kron(before,np.kron(_U,np.kron(uninvolved,np.kron(proj1,after))))
    else:
        raise Exception(f'Conditional Error: control and target are the same "{c} == {t}"')
    return a+b

print("dense:")
n=3
psi=np.kron(np.kron(_0[1], _0[1]),_0[1]) #all 0's state
print(f"psi:\n{psi}")
u=U(_H,3,n)                     #apply Hadamar on qubit 3
print(f"apply u:\n{u}")
psi = np.dot(psi,u)
print(f"psi:\n{psi}")
c=C(_X,3,1,n)                   #conditionally apply Not on qubit 1
print(f"apply c:\n{c}")
psi = np.dot(psi,c)
print(f"psi:\n{psi}")
print()

import quimb as qu
import quimb.tensor as qtn
print("tensor network:")
n=3
psi = qtn.tensor_builder.MPS_computational_state('0'*n) #all 0's state
print(f"psi:\n{psi.to_dense()}")
arrays = [_I, _I, _H]                                   #apply Hadamar on qubit 3
u=qtn.tensor_builder.MPO_product_operator(arrays)
print(f"apply u:\n{u}")
psi = u.apply(psi)
print(f"psi:\n{psi.to_dense()}")
#c=C(_X,3,1,n)                                          #conditionally apply Not on qubit 1
#transform or constuct u as MPO 
#c = qtn.MPO_rand(n,1) #placeholder
c = C(_X,3,1,n).reshape(16,2,2) #does not work
c = qtn.tensor_builder.MPO_product_operator(c)
print(f"apply c:\n{c}")
psi = c.apply(psi)
print(f"psi:\n{psi.to_dense()}")

