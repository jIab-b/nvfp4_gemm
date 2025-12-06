###### 9.7.16.6.4.1. Pipelined instructions, same thread 

In this pattern, no explicit ordering mechanism is needed and the ordering guarantee is
provided by the pipelined instruction pairing.

Example:

```
tcgen05.mma

tcgen05.mma (same shape and accumulator)
```

The two instructions will be executed in program order.