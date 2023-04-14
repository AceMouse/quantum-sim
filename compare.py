from dense.sim import interpret as dinterpret
from tensor_network.sim import interpret as tinterpret
import numpy as np
matrix = dinterpret("circuits/QFT8.out", silent=True)
maxes = []
for c in range(16,64):
    mpo = tinterpret("circuits/QFT8_g_l.out",silent=True,max_bond=c).to_dense()
    diffs = np.array([(x-y) for x,y in zip(matrix.flatten(), mpo.flatten())])
    m=np.sum(diffs*diffs.conjugate()).real
    print(f'max_bond({c}): {m}')
    maxes += [(m,c)] 
    if m < 5e-25:
        break

cutoffs = []
for i in range(16,64):
    c = 1/(2**i)
    mpo = tinterpret("circuits/QFT8_g_l.out",silent=True,cutoff=c).to_dense()
    diffs = np.array([(x-y) for x,y in zip(matrix.flatten(), mpo.flatten())])
    m=np.sum(diffs*diffs.conjugate()).real
    print(f'cutoff({c}: {m})')
    cutoffs += [(m,c)] 
    if m < 5e-25:
        break

import matplotlib.pyplot as plt

plt.plot([c for m,c in maxes], [m for m,c in maxes])
  
plt.xlabel('max_bond')
plt.ylabel('diff')
#plt.semilogx()
#plt.semilogy()
plt.title('max_bond')
  
plt.show()
plt.plot([c for m,c in cutoffs], [m for m,c in cutoffs])
  
plt.xlabel('cutoff')
plt.ylabel('diff')
plt.semilogx()
plt.semilogy()
  
plt.title('cutoff')
  
plt.show()

