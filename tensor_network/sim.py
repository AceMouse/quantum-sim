import numpy as np
import math
import cmath
import sys

import quimb as qu
import quimb.tensor as qtn

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
    return [_I]*k
#return a matrix that applies gate U to the t'th qubit in an n qubit state.
def U(_U, t, n): 
    proj0 = _0[1]*_0[0]
    proj1 = _1[1]*_1[0]
    before = I(t-1) 
    after = I(n-t)    
    return before+[_U]+after
    
# https://quantumcomputing.stackexchange.com/a/4255 <- math
#return a matrix that applies gate U to the t'th qubit, controled by the c'th qubit, in an n qubit state 
def C(_U, c, t, n): 
    raise Exception("multi qubit gates not implemented!")
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

def parse_state(state_string):
    return qtn.tensor_builder.MPS_computational_state(state_string)

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

def get_MPOs(path):
    with open(path, 'r') as out:
        instrs = out.read().splitlines()
    MPOs = []
    n = int(instrs[0])
    i = 1
    d = {'H':_H,'X':_X,'I':_I}
    while i < len(instrs):
        g = None
        is_cond = instrs[i] == 'C'
        i+= is_cond
        if instrs[i] in d:
            g = d[instrs[i]]
        elif instrs[i][0] == 'R':
            g = _R_n(int(instrs[i][1:]))
        i+= 1
        c = int(instrs[i])
        i+= is_cond
        t = int(instrs[i])
        i+= 1
        o = C(g,c,t,n) if is_cond else U(g,t,n)
        o = qtn.tensor_builder.MPO_product_operator(o)
        MPOs.append(o)
    return n, MPOs

def interpret(path, state_string, reverse = False, debug=False):
    state = parse_state(state_string)
    if debug:
        print("before:")
        print_state(state.to_dense())
        print_measurement(state)
    n, MPOs = get_MPOs(path)
    if reverse:
        MPOs = MPOs[::-1]
    for MPO in MPOs:
        state = MPO.apply(state)
    if debug:
        print("after:")
        print_state(state.to_dense())
    print_measurement(state)

if '-i' in sys.argv:
    interpret(sys.argv[1], sys.argv[2], reverse = '-r' in sys.argv, debug = '-d' in sys.argv)
