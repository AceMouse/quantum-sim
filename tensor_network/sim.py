import numpy as np
import math
import cmath
import sys
import scipy.linalg as la
from scipy.linalg  import svdvals, diagsvd

import quimb as qu
import quimb.tensor as qtn

_0     = [np.array([[1],[0]], dtype=complex), np.array([1,0], dtype=complex)]
_1     = [np.array([[0],[1]], dtype=complex), np.array([0,1], dtype=complex)]
_plus  = [(_0[i] + _1[i])/math.sqrt(2) for i in range(len(_0))]
_minus = [(_0[i] - _1[i])/math.sqrt(2) for i in range(len(_0))]

_I = np.array([[1,0],[0,1]], dtype=complex)
_X = np.array([[0,1],[1,0]], dtype=complex)
_H = (1/math.sqrt(2))*np.array([[1,1],[1,-1]], dtype=complex)
_S = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=complex)
_s = np.array([[1,0],[0,1j]], dtype=complex)
_Z = np.array([[1,0],[0,-1]], dtype=complex)
_Y = np.array([[0,-1j],[1j,0]], dtype=complex)

def kron(arr, dtype=complex):
    res = np.array([1],dtype=dtype)
    for x in arr:
        res = np.kron(res, x)
    return res

def _R(angle):
    return np.array([[1,0],[0,cmath.exp(1j*angle)]], dtype=complex)

def _R_n(n):
    return _R(2*math.pi/math.pow(2,n))

def I(k, begining=False, end=False):
    if k<1:
        return []
    if k==1:
        if begining and end:
            return [_I.reshape(2,2)]
        elif begining or end:
            return [_I.reshape(1,2,2)]
        return [_I.reshape(1,1,2,2)]
    return [_I.reshape(1,2,2) if begining else _I.reshape(1,1,2,2)] + ([_I.reshape(1,1,2,2)]*(k-2)) + [_I.reshape(1,2,2) if end else _I.reshape(1,1,2,2)]

def swapBits(x, p1, p2, n):
    set1 =  (x >> p1) & ((1<< n) - 1)
    set2 =  (x >> p2) & ((1 << n) - 1)
    xor = (set1 ^ set2)
    xor = (xor << p1) | (xor << p2)
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
    
def operator_schmidt_decomposition(_U, begining = False, end = False):
    A=transform(_U)
    u,s,vh = np.linalg.svd(A)
    ss=np.sqrt(s)
    ssd=la.diagsvd(ss,*A.shape)
    U=u@ssd
    Vh=ssd@vh
    if begining:
        U=U.transpose().reshape(4,2,2)
    else:
        U=U.transpose().reshape(1,4,2,2)
    if end:
        Vh=Vh.reshape(4,2,2)
    else:
        Vh=Vh.reshape(4,1,2,2)
    return [U,Vh]
#return a matrix that applies gate U to the t'th qubit in an n qubit state.
def U(_U, t, n): 
    before = I(t-1, begining=True) 
    after = I(n-t, end=True)    
    _U = _U.reshape(1,1,2,2)
    if t==1==n:
        _U = _U.reshape(2,2)
    elif t==1 or t==n:
        _U = _U.reshape(1,2,2)
    return before+[_U]+after
    
def printm(m):
    for i in m:
        for x in i:
            if x != 0j:
                if x.real >0:
                    print(' ', end='')
                print(f'{x:.2f} ',end = '')
            else:
                print('            ',end = '')
            print(" ",end = '')
        print()
# https://quantumcomputing.stackexchange.com/a/4255 <- math
#return a matrix that applies gate U to the t'th qubit, controled by the c'th qubit, in an n qubit state 
def U2(_U, t1, t2, n):
    if ((t1-t2)**2 > 1):
        raise Exception("non-local multi qubit gates not implemented!")
    if t1 == t2:
        raise Exception(f'Conditional Error: control and target are the same "{c} == {t}"')
    _min = min(t1,t2)
    _max = max(t1,t2)
    before = I(_min-1, begining=True)    
    after = I(n-_max, end=True)
    return before+operator_schmidt_decomposition(_U, begining=_min==1,end=_max==n)+after

def C(_U, c, t, n):
    if ((c-t)**2 > 1):
        raise Exception("non-local multi qubit gates not implemented!")
    if c == t:
        raise Exception(f'Conditional Error: control and target are the same "{c} == {t}"')
    proj0 = _0[1]*_0[0]
    proj1 = _1[1]*_1[0]
    x = [proj0, _I]
    y = [proj1, _U]
    if t < c:
        x = x[::-1]
        y = y[::-1]
    u = kron([x[0],x[1]]) + kron([y[0],y[1]])
    return U2(u,c,t,n) 

def parse_braket(dirac):
    ket = 1
    if len(dirac) > 2: 
        x,*y,z = dirac
        if x+z == '|>':
            dirac = y
        elif x+z == '<|':
            dirac = y
            ket = 0
    d = {'1':_1[ket], '0':_0[ket], '+':_plus[ket], '-':_minus[ket]}
    vec = []
    for bit in dirac:    
        if bit in d:
            vec.append(d[bit])
        else:
            raise Exception(f'Parse Error "{bit}"')
    vec[0] = vec[0].reshape(1,2)
    vec[-1] = vec[-1].reshape(1,2)
    if len(vec)>2:
        vec = [vec[0]]+[v.reshape(1,1,2) for v in vec[1:-1]]+[vec[-1]]
    return len(dirac), vec

def parse_state(state_string):
    x, t = parse_braket(state_string);
    return x, qtn.tensor_1d.MatrixProductState(t)

def print_state(state, shorten=True, postfix=''):
    state = state.flatten()
    (size,) = state.shape
    size -= 1
    print(f'{"#".rjust(len(str(size)))} {"state".ljust(size.bit_length()+2)} probability')
    for i,s in enumerate(state):
        if not shorten or s != 0:
            print(f'{str(i).rjust(len(str(size)))} |{(f"{i:b}>".rjust(size.bit_length()+1,"0")).ljust(5)} {s}{postfix}')

def measure(state):
    return np.abs(np.square(state.to_dense().flatten(), dtype=complex))

def print_measurement(state, shorten=True):
    print_state(np.around(measure(state)*100,decimals=1), shorten, postfix='%')

def get_MPO(path, max_bond=None, cutoff=None):
    with open(path, 'r') as out:
        instrs = out.read().splitlines()
    n = int(instrs[0])
    if n < 2:
        print("Tensor simpulater does not work for circuits of 1 qubit. Please provide another circuit.")
        quit()
    MPO = qtn.tensor_builder.MatrixProductOperator(I(n, begining=True, end=True))
    i = 1
    d = {'H':_H,'X':_X,'I':_I,'S':_S,'s':_s,'Z':_Z,'Y':_Y}
    while i < len(instrs):
        g = None
        is_cond = instrs[i] == 'C'
        is_2q = instrs[i] == 'S'
        i+= is_cond
        if instrs[i] in d:
            g = d[instrs[i]]
        elif instrs[i][0] == 'R':
            g = _R_n(int(instrs[i][1:]))
        i+= 1
        c = int(instrs[i])
        i+= is_cond + is_2q
        t = int(instrs[i])
        i+= 1
        o = C(g,c,t,n) if is_cond else (U2(g,c,t,n) if is_2q else U(g,t,n))
        o = qtn.tensor_builder.MatrixProductOperator(o)
        MPO=o.apply(MPO)
        if (max_bond != None or cutoff != None) and i%1==0:
            MPO.right_compress(max_bond=max_bond, cutoff=cutoff)
            MPO.left_compress(max_bond=max_bond, cutoff=cutoff)

    MPO.right_compress(max_bond=max_bond, cutoff=cutoff)
    MPO.left_compress(max_bond=max_bond, cutoff=cutoff)
    return n, MPO
import sys, os

def disablePrint():
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    sys.stdout = sys.__stdout__

def interpret(path, state_string='', reverse = False, debug=False, silent=False, max_bond=None, cutoff=None):
    if silent:
        disablePrint()
    if reverse:
        raise Exception('reverse not implemented yet')
    n, MPO = get_MPO(path, max_bond=max_bond, cutoff=cutoff)
    if state_string == '':
        printm(MPO.to_dense())
        enablePrint()
        return MPO
    x, state = parse_state(state_string)
    if n != x :
        print(f"Input of {x} qubits was provided. Please provide input state of {n} qubits for this circuit. ")
        quit()
    if debug:
        print("before:")
        print_state(state.to_dense())
        print_measurement(state)
    state = MPO.apply(state)
    if debug:
        print("after:")
        print_state(state.to_dense())
    if n<=16:
        print_measurement(state)
    enablePrint()
    return state

if __name__ == "__main__":
    r = '-r' in sys.argv
    d = '-d' in sys.argv
    i = '-i' in sys.argv
    s = '-s' in sys.argv
    b = '-b' in sys.argv
    if i and b:
        interpret(sys.argv[1], sys.argv[2], reverse = r, debug = d, silent = s, max_bond=int(sys.argv[3]))
    elif i:
        interpret(sys.argv[1], sys.argv[2], reverse = r, debug = d, silent = s)
    elif b:
        interpret(sys.argv[1], reverse = r, debug = d, silent = s, max_bond=int(sys.argv[2]))
    elif len(sys.argv) > 1:
        interpret(sys.argv[1], reverse = r, debug = d, silent = s)
        
