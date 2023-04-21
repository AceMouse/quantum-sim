import sys
import os
def gen(n, local_only=False):
    instrs = [n]
    n = int(n)
    if local_only:
        for step in range(1,n+1):
            instrs += ['H', step]
            swaps = []
            for i,x in enumerate(range(step,n)):
                s=['S', x+1, x]
                instrs += ['C',f'R{i+2}',x+1,x]
                if x < n-1: 
                    instrs += s
                    swaps += [s]
            for s in swaps[::-1]:
                instrs += s
    else:
        for step in range(n):
            instrs += ['H', step+1]
            for row in range(2,n-step+1):
                c=row+step
                t=step+1
                instrs += ['C',f'R{row}',c, t]

    out_path = f'QFT{n}_g{"_l"*local_only}.out'
    try:
        os.remove(out_path)
    except OSError:
        pass

    with open(out_path, 'a') as out:
        for instr in instrs:
            out.write(f'{instr}\n')
        

gen(sys.argv[1], local_only='-l' in sys.argv)

