## 8.7. Morally strong operations 

Two operations are said to be *morally strong* relative to each other if they satisfy all of the
following conditions:

1. The operations are related in *program order* (i.e, they are both executed by the same thread),
   or each operation is *strong* and specifies a *scope* that includes the thread executing the
   other operation.
2. Both operations are performed via the same *proxy*.
3. If both are memory operations, then they overlap completely.

Most (but not all) of the axioms in the memory consistency model depend on relations between
*morally strong* operations.

### 8.7.1. [Conflict and Data-races](#conflict-and-data-races)[](#conflict-and-data-races "Permalink to this headline")

Two *overlapping* memory operations are said to *conflict* when at least one of them is a *write*.

Two *conflicting* memory operations are said to be in a *data-race* if they are not related in
*causality order* and they are not *morally strong*.

### 8.7.2. [Limitations on Mixed-size Data-races](#mixed-size-limitations)[](#mixed-size-limitations "Permalink to this headline")

A *data-race* between operations that *overlap* completely is called a *uniform-size data-race*,
while a *data-race* between operations that *overlap* partially is called a *mixed-size data-race*.

The axioms in the memory consistency model do not apply if a PTX program contains one or more
*mixed-size data-races*. But these axioms are sufficient to describe the behavior of a PTX program
with only *uniform-size data-races*.

Atomicity of mixed-size RMW operations

In any program with or without *mixed-size data-races*, the following property holds for every pair
of *overlapping atomic* operations A1 and A2 such that each specifies a *scope* that includes the
other: Either the *read-modify-write* operation specified by A1 is performed completely before A2 is
initiated, or vice versa. This property holds irrespective of whether the two operations A1 and A2
overlap partially or completely.