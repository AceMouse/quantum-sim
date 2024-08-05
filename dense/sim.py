import numpy as np
import math
import cmath
import sys

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

def I(k):
    return kron([_I]*k)

#return a matrix that applies gate U to the t'th qubit in an n qubit state.
def U(_U, t, n): 
    before = I(t-1) 
    after = I(n-t)    
    return kron([before,_U,after])
    
def U2(_U, c, t, n): 
    before = I(min(c,t)-1) 
    after = I(n-max(c,t))    
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

def parse_braket(dirac):
    ket = 0
    if len(dirac) > 2: 
        x,*y,z = dirac
        if x+z == '|>':
            dirac = y
        elif x+z == '<|':
            dirac = y
            ket = 1
    d = {'1':_1[ket], '0':_0[ket], '+':_plus[ket], '-':_minus[ket]}
    vec = [[1]]
    for bit in dirac:    
        if bit in d:
            m = d[bit]
        else:
            raise Exception(f'Parse Error "{bit}"')
        vec = np.kron(vec, m)
    return len(dirac), vec
def parse_state(state_string):
    return parse_braket(state_string)

def print_state(state, shorten=True, postfix=''):
    state = state.flatten()
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

def combine(path):
    with open(path, 'r') as out:
        instrs = out.read().splitlines()
    n = int(instrs[0])
    m = I(n)
    i = 1
    d = {'H':_H,'X':_X,'I':_I,'S':_S,'s':_s,'Z':_Z,'Y':_Y}
    while i < len(instrs):
        g = None
        is_cond = instrs[i] == 'C'
        is_2q = instrs[i] == 'S'
#        if instrs[i] == 'H':
#            print('----------------------------')
#            printm(m)
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
#        printm(m)
        m = o@m
#        print('            *               ')
#        printm(o)
#        print('----------------------------')
#    printm(m)
    return n, m

import sys, os

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


def interpret(path, state_string='', reverse = False, debug=False, silent=False):
    if silent:
        blockPrint()
    n, m = combine(path)
    if reverse:
        m = m.conj().T
    if state_string == '':
        printm(m)
        enablePrint()
        return m
    x, state = parse_state(state_string)
    if n != x :
        print(f"Input of {x} qubits was provided. Please provide input state of {n} qubits for this circuit. ")
        enablePrint()
        quit()
    if debug:
        print("before:")
        print_state(state)
        print_measurement(state)
    state = m@state
    if debug:
        print("after:")
        print_state(state)
    print_measurement(state)
    #entropy = vn_entropy_from_state(state, n, reverse = reverse, debug = debug)
    #if not debug:
    #    print("Von Neumann entropy:")
    #    print(entropy)
    enablePrint()
    return state

#translated to python from https://github.com/jonasmaziero/LibForQ/blob/master/partial_trace.f90
def partial_trace_a(rho, da, db, debug=False):
    if debug:
        print(f"da {da}, db {db}")
        print(rho)
    rho_b = np.zeros((db, db), dtype=complex)
    if debug:
        print(f"da {da}, db {db}")
        print(rho)
    for j in range(db):
        for k in range(db):
            for l in range(da):
                rho_b[j, k] += rho[l * db + j, l * db + k]

    if debug:
        print(rho_b)
    return rho_b

#translated to python from https://github.com/jonasmaziero/LibForQ/blob/master/partial_trace.f90
def partial_trace_b(rho, da, db, debug=False):
    rho_a = np.zeros((da, da), dtype=complex)

    if debug:
        print(f"da {da}, db {db}")
        print(rho)
    for j in range(da):
        for k in range(da):
            for l in range(db):
                rho_a[j, k] += rho[j * db + l, k * db + l]
    if debug:
        print(rho_a)

    return rho_a

def partial_trace(n, rho, trace_out, debug=False):
    id = []
    last = 0
    s,t = trace_out
    k = n - (t-s+1)
    before = I(s)
    after = I(n-t-1) 
    rho_a = None
    if debug:
        print(1<<k)
        print(n, '\n', s, n-t)

    for i in range(1<<k):
        j = format(i,f"0{k}b")
        j = kron([before,parse_state(f'|{j}>'),after])
        if debug:
            print(i, '\n', j, '\n', rho)
        total = np.matmul(np.matmul(j, rho), j.conj().T)
        if rho_a is None:
            rho_a = total
        else:
            rho_a += total
    return rho_a
        

def vn_entropy_from_state(state, n, reverse = False, k = -1, debug = True):
    if k == -1:
        k = n>>1
    rho = np.outer(state, np.conj(state))
    rho_a = partial_trace_b(rho, n, n, debug=debug)
    if debug:
        print("rho_a: ")
        print(np.round(rho_a,4))
    return vn_entropy_from_partial_trace(rho_a, reverse = reverse, debug = debug)

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

def vn_entropy_from_circuit(path, reverse = False, k = -1, repeat = False, debug=False):
    n, m = combine(path)
    if reverse:
        m = m.conj().T
    z = 1
    if repeat:
        z = 1<<n
    max_entropy = -1000
    input = ''
    for x in range(z):
        psi = format(x,f'0{n}b')
        state = m@parse_state(f'|{psi}>')
        entropy = vn_entropy_from_state(state, n, reverse, k, debug = False) 
        if entropy > max_entropy: 
            max_entropy = entropy
            input = f'|{psi}>'
    print(f"max vn entropy (input {input}):")
    print(vn_entropy_from_state(m@parse_state(input), n, reverse, k, debug = debug) )

def trace_test(debug = False):
    print("____________________________________________________")
    print("old")
    print(partial_trace(2, np.array([[0,0,1,0],[1,0,0,0],[0,0,0,0],[0,0,0,0]]), (0,0), debug=debug))
    print(partial_trace(2, np.array([[0,0,1,0],[1,0,0,0],[0,0,0,0],[0,0,0,0]]), (1,1), debug=debug))

    print("____________________________________________________")
    print("new")
    print(partial_trace_a(np.array([[0,0,1,0],[1,0,0,0],[0,0,0,0],[0,0,0,0]], dtype=complex), 2,2, debug=debug))
    print(partial_trace_b(np.array([[0,0,1,0],[1,0,0,0],[0,0,0,0],[0,0,0,0]], dtype=complex), 2,2, debug=debug))

def vn_entropy_test(debug = False):
    print("vn entropy test:")
    print("____________________________________________________")
    print("QFT 2 qubits")
    vn_entropy_from_circuit('circuits/QFT2.out', repeat = True, debug=debug)
    print("____________________________________________________")
    print("QFT 4 qubits")
    vn_entropy_from_circuit('circuits/QFT4.out', repeat = True, debug=debug)
    print("____________________________________________________")
    print("QFT 8 qubits")
    vn_entropy_from_circuit('circuits/QFT8.out', repeat = True, debug=debug)
    print("____________________________________________________")
    print("Absolutely Maximal Entropy 2 qubits")
    vn_entropy_from_circuit('circuits/AME2.out', repeat = True, debug=debug)
    print("____________________________________________________")
    print("Absolutely Maximal Entropy 4 qubits")
    vn_entropy_from_circuit('circuits/AME4.out', repeat = True, debug=debug)
    print("____________________________________________________")
    print("Absolutely Maximal Entropy 6 qubits")
    vn_entropy_from_circuit('circuits/AME6.out', repeat = True, debug=debug)
    print("____________________________________________________")
    print("Identity 2 qubits")
    vn_entropy_from_circuit('circuits/I2.out', repeat = True, debug=debug)
    print("____________________________________________________")
    print("Identity 4 qubits")
    vn_entropy_from_circuit('circuits/I4.out', repeat = True, debug=debug)
    print("____________________________________________________")
    print("Identity 8 qubits")
    vn_entropy_from_circuit('circuits/I8.out', repeat = True, debug=debug)
    print("____________________________________________________")

if __name__ == '__main__':
    r = '-r' in sys.argv
    d = '-d' in sys.argv
    i = '-i' in sys.argv
    b = '-b' in sys.argv
    t = '-t' in sys.argv
    s = '-s' in sys.argv
    if i:
        interpret(sys.argv[1], state_string=sys.argv[2], reverse = r, debug = d, silent = s)
    if b:
        vn_entropy_test(debug = d)
    if t:
        trace_test(debug = d)
    if not b and not t and not i:
        interpret(sys.argv[1], debug = d, silent=s)
