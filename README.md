# quantum-sim
Simulating quantum-circuits in different ways

# compile & run
use flag `-r` for reverse aplication of the circuit
```
❯ cat circuits/QFT2
--H-R2--
----C-H-
❯ python3 compile/compiler.py circuits/QFT2
❯ cat circuits/QFT2.out
2
H
1
C
R2
2
1
H
2
❯ python3 dense/sim.py circuits/QFT2.out '|10>'
before:
# state probability
2 |10>   (1+0j)
# state probability
2 |10>   100.0%
after:
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
Von Neumann entropy:
(0.6931471805599454+0j)
❯ python3 dense/sim.py circuits/QFT2.out '|-+>' -r
before:
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
after:
# state probability
0 |00>   (-2.465190328815662e-32+0j)
2 |10>   (0.9999999999999996+0j)
# state probability
2 |10>   100.0%
Von Neumann entropy:
(0.6931471805599454+0j)
```
