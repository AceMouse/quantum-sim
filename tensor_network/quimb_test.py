import quimb.tensor as qtn
import math
import cmath
import numpy as np
import quimb as q
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
    m = []
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
            m += [op]
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
        m += [op]
        i += 2
    z = [qtn.tensor_builder.MPO_product_operator(x) for x in m]
    print(m)
    print()
    print(z)
    return n, z
def interpret(path, state_string, reverse = False, debug=False):
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
        state = np.dot(parse_state(f'|{psi}>'), m)
        entropy = vn_entropy_from_state(state, n, reverse, k, debug = False) 
        if entropy > max_entropy: 
            max_entropy = entropy
            input = f'|{psi}>'
    print(f"max vn entropy (input {input}):")
    print(vn_entropy_from_state(np.dot(parse_state(input), m), n, reverse, k, debug = debug) )

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

if '-i' in sys.argv:
    interpret(sys.argv[1], sys.argv[2], reverse = '-r' in sys.argv, debug = '-d' in sys.argv)
if '-b' in sys.argv:
    vn_entropy_test(debug = '-d' in sys.argv)
#_q0 = qtn.Tensor(q.qu(_0[1]), inds=('k0', 'b0'), tags=['0'])
#_q1 = qtn.Tensor(q.qu(_1[1]), inds=('k1', 'b1'), tags=['1'])
'''
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
'''
def QFT(n):
    gates = []
    for i in range(n):
        gates+=[U(_H,i,n)]
        for x,j in enumerate(range(i,n-1)):
            gates+=[U(_R_n(x+1),i,j+1)]
    MPOs = [qtn.tensor_builder.MPO_product_operator(x) for x in gates]
    return MPOs 

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
def main1():    
    n=8
    qc = qtn.Circuit(n)
    qc.apply_gates(prepare(input()))
    print("PREPARED STATE:")
    print_psi(qc)
    qc.apply_gates(QFT(n))
    print("OUTPUT STATE:")
    print_psi(qc)
    print(qc.psi.draw(color=['PSI0']))
    print(qc.psi^...)
    print((qc.psi^...).draw(color=['PSI0']))
    #print(qc.psi.to_dense().round(4))
    rho_a = qc.partial_trace((0,int(n/2))).round(4)
    print(type(rho_a))
    print("TRACE:")
    print(rho_a)
    vn_entropy_from_partial_trace(rho_a, debug=True)
    #print(qc.amplitude(f'{2:b}'.ljust(n,'0')))

def main():
    n=4
    states = '0'*n
    psi = qtn.tensor_builder.MPS_computational_state(states)
    print(psi)
    print(psi.to_dense())
    print()
    for i in QFT(n):
        print(i)


main()
