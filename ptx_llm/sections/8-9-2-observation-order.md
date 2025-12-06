### 8.9.2. Observation Order 

*Observation order* relates a write W to a read R through an optional sequence of atomic
read-modify-write operations.

A write W precedes a read R in *observation order* if:

1. R and W are *morally strong* and R reads the value written by W, or
2. For some atomic operation Z, W precedes Z and Z precedes R in *observation order*.