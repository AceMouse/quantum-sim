import sys
import re
import os
def T(l):
    return [list(x) for x in zip(*l)]

def compile(path):
    tokens = []
    with open(path, 'r') as circ:
        lines = circ.readlines()
        for i,line in enumerate(lines):
            tokens.append(re.findall('..',line))
    instructions = []
    for col in T(tokens):
        c = None
        t = None
        U = None
        for qubit, token in enumerate(col):
            if token == '--':
                continue
            if token == 'C-':
                c = qubit
                continue
            m = re.search('R\d',token)
            if not m is None:
                U = m.group()
                t = qubit
                continue
            m = re.search('[HX]-',token)
            if not m is None:
                U = m.group()[0]
                t = qubit
                continue
        if U is None: 
            continue
        if not c is None:
            instructions.append('C')
        instructions.append(U)
        if not c is None:
            instructions.append(c+1)
        instructions.append(t+1)

    out_path = path + '.out'
    try:
        os.remove(out_path)
    except OSError:
        pass

    with open(out_path, 'a') as out:
        for instr in instructions:
            out.write(f'{instr}\n')
        

compile(sys.argv[1])
