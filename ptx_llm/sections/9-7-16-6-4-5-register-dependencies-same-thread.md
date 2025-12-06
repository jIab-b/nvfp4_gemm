###### 9.7.16.6.4.5. Register dependencies, same thread 

For `tcgen05.ld`, an intra-thread ordering through true register dependency will be respected
regardless of the presence or absence of other forms of synchronization. This form of register
dependency does not imply any other form of ordering. For example, a register dependency does
not imply that a dependee instruction’s memory accesses will be performed before a dependent
instruction’s memory accesses. To enforce such memory orderings and avoiding anti-dependency
hazards around `tcgen05.ld`, `tcgen05.wait::ld` must be used.

Example:

```
tcgen05.ld %r1, ...;

tcgen05.mma ..., %r1, ...;
```