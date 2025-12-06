### 8.9.7. Communication Order 

The *communication order* is a non-transitive order, determined at runtime, that relates write
operations to other *overlapping* memory operations.

1. A write W precedes an *overlapping* read R in *communication order* if R returns the value of any
   byte that was written by W.
2. A write W precedes a write W’ in *communication order* if W precedes W’ in *coherence order*.
3. A read R precedes an *overlapping* write W in *communication order* if, for any byte accessed by
   both R and W, R returns the value written by a write W’ that precedes W in *coherence order*.

*Communication order* captures the visibility of memory operations — when a memory operation X1
precedes a memory operation X2 in *communication order*, X1 is said to be visible to X2.