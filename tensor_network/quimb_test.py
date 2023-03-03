import quimb.tensor as qtn
import math
import cmath
import numpy as np
import quimb as q
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

def _CR_n(n):
    proj0 = _0[1]*_0[0]
    proj1 = _1[1]*_1[0]
    return q.qu(np.kron(proj0,_I)+np.kron(proj1,_R_n(n)))

def vn_entropy_from_partial_trace(rho_a, reverse = False, debug = True):
    # -sum_k(lambda_k ln lambda_k)
    eigen_values, eigen_vectors = np.linalg.eig(rho_a)
    if debug:
        print("eigen values (sorted): ")
        print(np.round(sorted(eigen_values, reverse =True), 4))
    entropy = 0
    for e in eigen_values:
        entropy -= e * np.log2(e + 1e-10)
    if debug:
        print("Von Neumann entropy: ")
        print(entropy)
    return entropy
_q0 = qtn.Tensor(q.qu(_0[1]), inds=('k0', 'b0'), tags=['0'])
_q1 = qtn.Tensor(q.qu(_1[1]), inds=('k1', 'b1'), tags=['1'])
n=8

qc = qtn.Circuit(n)

#print(qc.psi)
def QFT(n):
    gates = []
    for i in range(n-1):
        gates+=[('H',i)]
        for x,j in enumerate(range(i,n-1)):
            gates+=[(_CR_n(x+1),j,j+1),('SWAP',j,j+1)]
        gates = gates[:-1]
        for j in range(n-2,i,-1):
            gates+=[('SWAP',j-1,j)]
    gates+=[('H',n-1)]
    return gates
#print(gates)
def prepare(psi0):
    gates = []
    for i,b in enumerate(psi0):
        if b=='1':
            gates +=[('X', i)]
        if b=='+':
            gates +=[('H', i)]
        if b=='-':
            gates +=[('X', i),('H', i)]

    return gates

def print_psi(qc):
    for i,p in enumerate(qc.psi.to_dense().round(4)):
        if (p != 0j):
            print(f'{i:b}'.rjust(n,'0'),abs(p[0])**2*100,'%  ',p[0])
qc.apply_gates(prepare(input()))
print_psi(qc)
qc.apply_gates(QFT(n))
print_psi(qc)
#print(qc.psi.draw(color=['PSI0']))
#print(qc.psi.to_dense().round(4))
rho_a = qc.partial_trace((0,int(n/2))).round(4)
print(rho_a)
vn_entropy_from_partial_trace(rho_a, debug=True)
#print(qc.amplitude(f'{2:b}'.ljust(n,'0')))
