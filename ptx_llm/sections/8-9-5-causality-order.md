### 8.9.5. Causality Order 

*Causality order* captures how memory operations become visible across threads through synchronizing
operations. The axiom “Causality” uses this order to constrain the set of write operations from
which a read operation may read a value.

Relations in the *causality order* primarily consist of relations in *Base causality order*1 , which is a transitive order, determined at runtime.

Base causality order

An operation X precedes an operation Y in *base causality order* if:

1. X precedes Y in *program order*, or
2. X *synchronizes* with Y, or
3. For some operation Z,

   1. X precedes Z in *program order* and Z precedes Y in *base causality order*, or
   2. X precedes Z in *base causality order* and Z precedes Y in *program order*, or
   3. X precedes Z in *base causality order* and Z precedes Y in *base causality order*.

Proxy-preserved base causality order

A memory operation X precedes a memory operation Y in *proxy-preserved base causality order* if X
precedes Y in *base causality order*, and:

1. X and Y are performed to the same address, using the *generic proxy*, or
2. X and Y are performed to the same address, using the same *proxy*, and by the same thread block,
   or
3. X and Y are aliases and there is an alias *proxy fence* along the base causality path from X
   to Y.

Causality order

*Causality order* combines *base causality order* with some non-transitive relations as follows:

An operation X precedes an operation Y in *causality order* if:

1. X precedes Y in *proxy-preserved base causality order*, or
2. For some operation Z, X precedes Z in observation order, and Z precedes Y in *proxy-preserved
   base causality order*.

1 The transitivity of *base causality order* accounts for the “cumulativity” of synchronizing
operations.