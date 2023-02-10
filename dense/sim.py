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
def _R(angle):
    return np.array([[1,0],[0,cmath.exp(1j*angle)]], dtype=complex)

def _R_n(n):
    return _R(2*math.pi/math.pow(2,n))

def I(k):
    I_n = np.array([1],dtype=complex)
    for i in range(k):
        I_n = np.kron(I_n,_I)
    return I_n
#return a matrix that applies gate U to the t'th qubit in an n qubit state.
def U(_U, t, n): 
    proj0 = _0[1]*_0[0]
    proj1 = _1[1]*_1[0]
    before = I(t-1) 
    after = I(n-t)    
    return np.kron(before,np.kron(_U,after))
    
# https://quantumcomputing.stackexchange.com/a/4255 <- math
#return a matrix that applies gate U to the t'th qubit, controled by the c'th qubit, in an n qubit state 
def C(_U, c, t, n): 
    proj0 = _0[1]*_0[0]
    proj1 = _1[1]*_1[0]
    _min = min(c,t)
    _max = max(c,t)
    before = I(_min-1)    
    uninvolved = I(_max-_min-1)
    after = I(n-_max)
    if c < t:
        a = np.kron(before,np.kron(proj0,np.kron(uninvolved,np.kron(_I,after))))
        b = np.kron(before,np.kron(proj1,np.kron(uninvolved,np.kron(_U,after))))
    elif t < c:
        a = np.kron(before,np.kron(_I,np.kron(uninvolved,np.kron(proj0,after))))
        b = np.kron(before,np.kron(_U,np.kron(uninvolved,np.kron(proj1,after))))
    else:
        raise Exception(f'Conditional Error: control and target are the same "{c} == {t}"')
    return a+b

def parse_braket(dirac):
    if len(dirac) > 2 and dirac[0] == '|' and dirac[-1] == '>':
        dirac = dirac[1:-1]
        ket = 1
    elif len(dirac) > 2 and dirac[0] == '<' and dirac[-1] == '|':
        dirac = dirac[1:-1]
        ket = 0
    else:
        ket = 1
        
    vec = None
    for bit in dirac:    
        if bit == '1':
            m = _1[ket]
        elif bit == '0':
            m = _0[ket]
        elif bit == '+':
            m = _plus[ket]
        elif bit == '-':
            m = _minus[ket]
        else:
            raise Exception(f'Parse Error "{bit}"')

        if vec is None:
            vec = m
        else: 
            vec = np.kron(vec, m)

    return np.atleast_2d(vec)

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

def combine(path):
    with open(path, 'r') as out:
        instrs = out.read().splitlines()
    m = 1
    n = int(instrs[0])
    i = 1
    while i < len(instrs):
        g = None
        if instrs[i] == 'C':
            if instrs[i+1] == 'H':
                g = _H
            elif instrs[i+1] == 'X':
                g = _X
            elif instrs[i+1] == 'I':
                g = _I
            elif instrs[i+1][0] == 'R':
                g = _R_n(int(instrs[i+1][1:]))
            c = int(instrs[i+2])
            t = int(instrs[i+3])
            op = C(g,c,t,n) 
            m = np.dot(m, op)
            i+=4
            continue
        
        if instrs[i] == 'H':
            g = _H
        elif instrs[i] == 'X':
            g = _X
        elif instrs[i] == 'I':
            g = _I
        elif instrs[i][0] == 'R':
            g = _R_n(int(instrs[i][1:]))
        t = int(instrs[i+1])
        op = U(g,t,n)
        m = np.dot(m, op)
        i += 2
    return n, m

def interpret(path, state_string, reverse = False, debug=False):
    state = parse_state(state_string)
    if debug:
        print("before:")
        print_state(state)
        print_measurement(state)
    n, m = combine(path)
    if reverse:
        m = m.conj().T
    state = np.dot(state, m)
    if debug:
        print("after:")
        print_state(state)
    print_measurement(state)
    entropy = vn_entropy_from_state(state, n, reverse = reverse, debug = debug)
    if not debug:
        print("Von Neumann entropy:")
        print(entropy)

#translated to python from https://github.com/jonasmaziero/LibForQ/blob/master/partial_trace.f90
def partial_trace_a(rho, da, db):
    rho_b = np.zeros((db, db), dtype=complex)

    for j in range(db):
        for k in range(db):
            for l in range(da):
                rho_b[j, k] += rho[l * db + j, l * db + k]

    return rho_b

#translated to python from https://github.com/jonasmaziero/LibForQ/blob/master/partial_trace.f90
def partial_trace_b(rho, da, db):
    rho_a = np.zeros((da, da), dtype=complex)

    for j in range(da):
        for k in range(da):
            for l in range(db):
                rho_a[j, k] += rho[j * db + l, k * db + l]

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
        j = np.kron(before, np.kron(parse_state(f'|{j}>'), after))
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
    if debug:
        print(rho)
    rho_a = partial_trace_b(rho, n, n)
    if debug:
        print("rho_a: ")
        print(np.round(rho_a,4))
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


def vn_entropy_from_circuit(path, reverse = False, k = -1, repeat = False):
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
        state = np.dot(parse_state(f'|{psi}>'), m)
        entropy = vn_entropy_from_state(state, n, reverse, k, debug = False) 
        if entropy > max_entropy: 
            max_entropy = entropy
            input = f'|{psi}>'
    print(f"max vn entropy (input {input}):")
    vn_entropy_from_state(np.dot(parse_state(input), m), n, reverse, k, debug = True) 

def trace_test():
    print(partial_trace(2, np.array([[0,0,1,0],[1,0,0,0],[0,0,0,0],[0,0,0,0]]), (0,0)))
    print(partial_trace(2, np.array([[0,0,1,0],[1,0,0,0],[0,0,0,0],[0,0,0,0]]), (1,1)))
    print(partial_trace_a(np.array([[0,0,1,0],[1,0,0,0],[0,0,0,0],[0,0,0,0]], dtype=complex), 2,2))
    print(partial_trace_b(np.array([[0,0,1,0],[1,0,0,0],[0,0,0,0],[0,0,0,0]], dtype=complex), 2,2))

def vn_entropy_test():
    print("vn entropy test:")
    print("____________________________________________________")
    print("QFT 4 qubits")
    vn_entropy_from_circuit('circuits/QFT4.out', repeat = True)
    print("____________________________________________________")
    print("QFT 8 qubits")
    vn_entropy_from_circuit('circuits/QFT8.out', repeat = True)
    print("____________________________________________________")
    print("QFT 2 qubits")
    vn_entropy_from_circuit('circuits/QFT2.out', repeat = True)
    print("____________________________________________________")
    print("Absolutely Maximal Entropy 4 qubits")
    vn_entropy_from_circuit('circuits/AME4.out', repeat = True)
    print("____________________________________________________")
    print("Identity 4 qubits")
    vn_entropy_from_circuit('circuits/I4.out', repeat = True)
    print("____________________________________________________")

if '-i' in sys.argv:
    interpret(sys.argv[1], sys.argv[2], reverse = '-r' in sys.argv, debug = '-d' in sys.argv)
if '-b' in sys.argv:
    vn_entropy_test()
if '-t' in sys.argv:
    trace_test()
