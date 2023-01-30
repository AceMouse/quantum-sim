# quantum-sim
Simulating quantum-circuits in different ways

# compile
```
❯ python3 compile/compiler.py cirquits/QFT2
❯ cat cirquits/QFT2.out
H
1
C
R2
2
1
H
2
❯ python dense/sim.py cirquits/QFT2.out 10
# state probability
0 |00>   (0.4999999999999999+0j)
1 |01>   (0.4999999999999999+0j)
2 |10>   (-0.4999999999999999+0j)
3 |11>   (-0.4999999999999999+0j)
# state probability
0 |00>   25.0%
1 |01>   25.0%
2 |10>   25.0%
3 |11>   25.0%
```
