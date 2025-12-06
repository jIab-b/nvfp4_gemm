### 8.9.6. Coherence Order 

There exists a partial transitive order that relates *overlapping* write operations, determined at
runtime, called the *coherence order*1. Two *overlapping* write operations are related in
*coherence order* if they are *morally strong* or if they are related in *causality order*. Two
*overlapping* writes are unrelated in *coherence order* if they are in a *data-race*, which gives
rise to the partial nature of *coherence order*.

1 *Coherence order* cannot be observed directly since it consists entirely of write
operations. It may be observed indirectly by its use in constraining the set of candidate
writes that a read operation may read from.