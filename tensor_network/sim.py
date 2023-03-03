import quimb.tensor as qtn
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

def C(g,c,t,n):
    if (g == 'H'):
        throw Exception("not implemented")
    if (g == 'X'):
        return ('CNOT', c, t)
    if (g == 'I'):
        return (,)
    if ( g== 'R'):

            


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

