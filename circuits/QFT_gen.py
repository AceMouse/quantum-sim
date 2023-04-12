import sys
import os
def gen(n, local_only=False):
    instrs = [n]
    n = int(n)
    for step in range(n):
        instrs += ['H', step+1]
        for row in range(2,n-step+1):
            swaps = []
            c=row+step
            t=step+1
            if local_only:
                while (c>(t+1)):
                    swaps += [['S', c, c-1]]
                    c-=1
                for s in swaps:
                    instrs += s
            instrs += ['C',f'R{row}',c, t]
            for s in swaps[::-1]:
                instrs += s

    out_path = f'QFT{n}_g{"_l"*local_only}.out'
    try:
        os.remove(out_path)
    except OSError:
        pass

    with open(out_path, 'a') as out:
        for instr in instrs:
            out.write(f'{instr}\n')
        

gen(sys.argv[1], local_only='-l' in sys.argv)

