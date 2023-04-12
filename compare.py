from dense.sim import interpret as dinterpret
from tensor_network.sim import interpret as tinterpret
import numpy as np
matrix = dinterpret("circuits/QFT8.out", silent=True)
mpos = [(tinterpret("circuits/QFT8_g_l.out",silent=True,max_bond=16),1) ] 
#mpos = [(tinterpret("circuits/QFT2.out",silent=True,max_bond=i),i) for i in range(2,33)] 
#print(dstate)
#print(tstate.to_dense())
maxes = []
for mpo,c in mpos:
    diffs = np.array([(x-y) for x,y in zip(matrix.flatten(), mpo.to_dense().flatten())])
    maxes += [(np.sum(diffs*diffs.conjugate()).real,c)] 

mpos = [(tinterpret("circuits/QFT8_g_l.out",silent=True,cutoff=10**-10),1) ] 
#mpos = [(tinterpret("circuits/QFT8_g_l.out", silent=True,cutoff=1/(2**i)),1/(2**i)) for i in range(10,30)]
#print(dstate)
#print(tstate.to_dense())
cutoffs = []
for mpo,c in mpos:
    diffs = np.array([(x-y) for x,y in zip(matrix.flatten(), mpo.to_dense().flatten())])
    cutoffs += [(np.sum(diffs*diffs.conjugate()).real,c)] 

print(cutoffs)
print(maxes)
'''
import matplotlib.pyplot as plt

plt.plot([c for m,c in maxes], [m for m,c in maxes])
  
plt.xlabel('max_bond')
plt.ylabel('diff')
plt.semilogx()
plt.semilogy()
plt.title('max_bond')
  
plt.show()
plt.plot([c for m,c in cutoffs], [m for m,c in cutoffs])
  
plt.xlabel('cutoff')
plt.ylabel('diff')
plt.semilogx()
plt.semilogy()
  
plt.title('cutoff')
  
plt.show()

for m,c in maxes:
    print(f'{m:.3f}: max_bond({c})')
for m,c in cutoffs:
    print(f'{m:.3f}: cutoff({c})')
'''
