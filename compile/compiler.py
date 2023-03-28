import sys
import re
import os
def T(l):
    return [list(x) for x in zip(*l)]

def compile(path, local_only=False):
    tokens = []
    with open(path, 'r') as circ:
        lines = circ.readlines()
        for i,line in enumerate(lines):
            tokens.append(re.findall('..',line))
    instructions = [f'{len(tokens)}']
    for col in T(tokens):
        c = None
        s1 = None
        t = None
        U = None
        for qubit, token in enumerate(col):
            if token == '--':
                continue
            if token == 'C-':
                c = qubit
                continue
            if token == 'S-':
                if s1 is None:
                    s1 = qubit
                    continue
                else:
                    U = 'S'
                    t=qubit
                    continue
            m = re.search('R\d',token)
            if not m is None:
                U = m.group()
                t = qubit
                continue
            m = re.search('[IHX]-',token)
            if not m is None:
                U = m.group()[0]
                t = qubit
                continue
        if U is None: 
            continue
        swaps = []
        if local_only and c:
            while c<(t-1)<t:
                swaps.append(['S',c+1,c+2])
                c+=1
            while c>(t+1)>t:
                swaps.append(['S',c+1,c])
                c-=1
        for s in swaps:
            instructions += s
        if not c is None:
            instructions.append('C')
        instructions.append(U)
        if not c is None:
            instructions.append(c+1)
        if not s1 is None:
            instructions.append(s1+1)
        instructions.append(t+1)
        for s in swaps[::-1]:
            instructions += s

    out_path = f'{path}{"_l"*local_only}.out'
    try:
        os.remove(out_path)
    except OSError:
        pass

    with open(out_path, 'a') as out:
        for instr in instructions:
            out.write(f'{instr}\n')
        

compile(sys.argv[1], local_only='-l' in sys.argv)
