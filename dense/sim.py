import numpy as np
import math
import cmath

_0     = np.array([1,0], dtype=complex)
_1     = np.array([0,1], dtype=complex)
_plus  = (_0 + _1)/math.sqrt(2)
_minus = (_0 - _1)/math.sqrt(2)

I = np.eye(2, dtype=complex)
X = np.array([[0,1],[1,0]], dtype=complex)
H = (1/math.sqrt(2))*np.array([[1,1],[1,-1]], dtype=complex)
def R(angle):
    return np.array([[1,0],[0,cmath.exp(1j*angle)]], dtype=complex)

def R_n(n):
    return R(2*math.pi/math.pow(2,n))

def parse_state(state_string):
    state = None
    for bit in state_string:    
        if bit == '1':
            m = _1
        elif bit == '0':
            m = _0
        elif bit == '+':
            m = _plus
        elif bit == '-':
            m = _minus
        else:
            raise Exception(f'Parse Error "{bit}"')

        if state is None:
            state = m
        else: 
            state = np.kron(state, m)

    return state

def print_state(state, shorten=True, postfix=''):
    (size,) = state.shape
    size -= 1
    print(f'{"#".rjust(len(str(size)))} {"state".ljust(size.bit_length()+2)} probability')
    for i,s in enumerate(state):
        if not shorten or s != 0:
            print(f'{str(i).rjust(len(str(size)))} |{(f"{i:b}>".rjust(size.bit_length()+1,"0")).ljust(5)} {s}{postfix}')

def measure(state):
    return np.abs(np.square(state, dtype=complex))

def print_measurement(state, shorten=True):
    print_state(np.around(measure(state)*100,decimals=1), shorten, postfix='%')

for n in range(10):
    state = np.dot(parse_state("1"),R_n(n))
    print_measurement(state)
    print_state(state)


state_string = input()
state = parse_state(state_string)
print_state(state)
print_measurement(state)

