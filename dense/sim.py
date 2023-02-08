import numpy as np
import math
import cmath
import sys

_0     = [np.array([[1],[0]], dtype=complex), np.array([1,0], dtype=complex)]
_1     = [np.array([[0],[1]], dtype=complex), np.array([0,1], dtype=complex)]
_plus  = [(_0[i] + _1[i])/math.sqrt(2) for i in range(len(_0))]
_minus = [(_0[i] - _1[i])/math.sqrt(2) for i in range(len(_0))]

I = np.array([[1,0],[0,1]], dtype=complex)
X = np.array([[0,1],[1,0]], dtype=complex)
H = (1/math.sqrt(2))*np.array([[1,1],[1,-1]], dtype=complex)
def R(angle):
    return np.array([[1,0],[0,cmath.exp(1j*angle)]], dtype=complex)

def R_n(n):
    return R(2*math.pi/math.pow(2,n))

#return a matrix that applies gate U to the t'th qubit in an n qubit state.
def U(U, t, n): 
    proj0 = _0[1]*_0[0]
    proj1 = _1[1]*_1[0]
    before = np.array([1],dtype=complex)
    for i in range(t-1):
        before = np.kron(before,I)
    after = np.array([1],dtype=complex)
    for i in range(n-t):
        after = np.kron(after,I)
    return np.kron(before,np.kron(U,after))
    
# https://quantumcomputing.stackexchange.com/a/4255 <- math
#return a matrix that applies gate U to the t'th qubit, controled by the c'th qubit, in an n qubit state 
def C(U, c, t, n): 
    proj0 = _0[1]*_0[0]
    proj1 = _1[1]*_1[0]
    _min = min(c,t)
    _max = max(c,t)
    before = np.array([1],dtype=complex)
    for i in range(_min-1):
        before = np.kron(before,I)
    uninvolved = np.array([1],dtype=complex)
    for i in range(_max-_min-1):
        uninvolved = np.kron(uninvolved,I)
    after = np.array([1],dtype=complex)
    for i in range(n-_max):
        after = np.kron(after,I)
    if c < t:
        a = np.kron(before,np.kron(proj0,np.kron(uninvolved,np.kron(I,after))))
        b = np.kron(before,np.kron(proj1,np.kron(uninvolved,np.kron(U,after))))
    elif t < c:
        a = np.kron(before,np.kron(I,np.kron(uninvolved,np.kron(proj0,after))))
        b = np.kron(before,np.kron(U,np.kron(uninvolved,np.kron(proj1,after))))
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
                g = H
            elif instrs[i+1] == 'X':
                g = X
            elif instrs[i+1] == 'I':
                g = I
            elif instrs[i+1][0] == 'R':
                g = R_n(int(instrs[i+1][1:]))
            c = int(instrs[i+2])
            t = int(instrs[i+3])
            op = C(g,c,t,n) 
            m = np.dot(m, op)
            i+=4
            continue
        
        if instrs[i] == 'H':
            g = H
        elif instrs[i] == 'X':
            g = X
        elif instrs[i] == 'I':
            g = I
        elif instrs[i][0] == 'R':
            g = R_n(int(instrs[i][1:]))
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

def vn_entropy_from_state(state, n, reverse = False, k = -1, debug = True):
    psi_ket = state
    psi_bra = psi_ket.conj().T
    if k == -1:
        k = n>>1
    before = np.array([1],dtype=complex)
    for i in range(n-k):
        before = np.kron(before,I)

    # sum_j_k+1.. =0 -> 1 (<I^k|<j_k+1|...<j_n|)|psi><psi|(|I^k>|j_k+1>...|j_n>)
    roh_a = None 
    for x in range(1<<k):
        j = format(x,f'0{k}b')
        jket = np.kron(before, parse_state(f'|{j}>'))
        jbra = jket.conj().T
        #print(jbra)
        #print(psi_ket)
        #print(psi_bra)
        #print(jket)
        left = np.matmul(jbra.conj().T, psi_ket.conj().T)
        right = np.matmul(psi_bra.conj().T, jket.conj().T)
        total = np.dot(left, right)
        #print(f'<{j}|00><00|{j}> = \n{total}')
        if roh_a is None:
            roh_a = total
        else:
            roh_a += total
    if debug:
        print("rho_a: ")
        print(np.round(roh_a,4))
    # -sum_k(lambda_k ln lambda_k)
    eigen_values, eigen_vectors = np.linalg.eig(roh_a)
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
