import numpy as np
_0 = np.array([[1],[0]], dtype=complex)
_1 = np.array([[0],[1]], dtype=complex)
def parse_state(state_string):
    state = None
    for bit in state_string:    
        if bit == '1':
            m = _1
        elif bit == '0':
            m = _0
        else:
            raise Exception(f'Parse Error "{bit}"')

        if state is None:
            state = m
        else: 
            state = np.kron(state, m)

    return state

def print_state(state, shorten=True, postfix=''):
    size, _ = state.shape
    size -= 1
    print(f'{"#".rjust(len(str(size)))} {"state".ljust(size.bit_length()+2)} probability')
    for i,s in enumerate(state[:,0]):
        if not shorten or s != 0:
            print(f'{str(i).rjust(len(str(size)))} |{(f"{i:b}>".rjust(size.bit_length()+1,"0")).ljust(5)} {s}{postfix}')

def measure(state):
    return np.square(state)

def print_measurement(state, shorten=True):
    print_state(measure(state).real*100, shorten, postfix='%')




state_string = input()
state = parse_state(state_string)
print_state(state)
print_measurement(state)
