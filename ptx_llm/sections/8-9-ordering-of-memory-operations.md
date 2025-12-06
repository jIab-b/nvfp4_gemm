## 8.9. Ordering of memory operations 

The sequence of operations performed by each thread is captured as *program order* while *memory
synchronization* across threads is captured as *causality order*. The visibility of the side-effects
of memory operations to other memory operations is captured as *communication order*. The memory
consistency model defines contradictions that are disallowed between communication order on the one
hand, and *causality order* and *program order* on the other.

### 8.9.1. [Program Order](#program-order)[](#program-order "Permalink to this headline")

The *program order* relates all operations performed by a thread to the order in which a sequential
processor will execute instructions in the corresponding PTX source. It is a transitive relation
that forms a total order over the operations performed by the thread, but does not relate operations
from different threads.

#### 8.9.1.1. [Asynchronous Operations](#program-order-async-operations)[](#program-order-async-operations "Permalink to this headline")

Some PTX instructions (all variants of `cp.async`, `cp.async.bulk`, `cp.reduce.async.bulk`,
`wgmma.mma_async`) perform operations that are asynchronous to the thread that executed the
instruction. These asynchronous operations are ordered after prior instructions in the same thread
(except in the case of `wgmma.mma_async`), but they are not part of the program order for that
thread. Instead, they provide weaker ordering guarantees as documented in the instruction
description.

For example, the loads and stores performed as part of a `cp.async` are ordered with respect to
each other, but not to those of any other `cp.async` instructions initiated by the same thread,
nor any other instruction subsequently issued by the thread with the exception of
`cp.async.commit_group` or `cp.async.mbarrier.arrive`. The asynchronous mbarrier [arrive-on](#parallel-synchronization-and-communication-instructions-mbarrier-arrive-on) operation
performed by a `cp.async.mbarrier.arrive` instruction is ordered with respect to the memory
operations performed by all prior `cp.async` operations initiated by the same thread, but not to
those of any other instruction issued by the thread. The implicit mbarrier [complete-tx](#parallel-synchronization-and-communication-instructions-mbarrier-complete-tx-operation)
operation that is part of all variants of `cp.async.bulk` and `cp.reduce.async.bulk`
instructions is ordered only with respect to the memory operations performed by the same
asynchronous instruction, and in particular it does not transitively establish ordering with respect
to prior instructions from the issuing thread.

### 8.9.2. [Observation Order](#observation-order)[](#observation-order "Permalink to this headline")

*Observation order* relates a write W to a read R through an optional sequence of atomic
read-modify-write operations.

A write W precedes a read R in *observation order* if:

1. R and W are *morally strong* and R reads the value written by W, or
2. For some atomic operation Z, W precedes Z and Z precedes R in *observation order*.

### 8.9.3. [Fence-SC Order](#fence-sc-order)[](#fence-sc-order "Permalink to this headline")

The *Fence-SC* order is an acyclic partial order, determined at runtime, that relates every pair of
*morally strong fence.sc* operations.

### 8.9.4. [Memory synchronization](#memory-synchronization)[](#memory-synchronization "Permalink to this headline")

Synchronizing operations performed by different threads synchronize with each other at runtime as
described here. The effect of such synchronization is to establish *causality order* across threads.

1. A `fence.sc` operation X *synchronizes* with a `fence.sc` operation Y if X precedes Y in the
   *Fence-SC* order.
2. A `bar{.cta}.sync` or `bar{.cta}.red` or `bar{.cta}.arrive` operation *synchronizes* with a
   `bar{.cta}.sync` or `bar{.cta}.red` operation executed on the same barrier.
3. A `barrier.cluster.arrive` operation synchronizes with a `barrier.cluster.wait` operation.
4. A *release* pattern X *synchronizes* with an *acquire* pattern Y, if a *write* operation in X
   precedes a *read* operation in Y in *observation order*, and the first operation in X and the
   last operation in Y are *morally strong*.

API synchronization

A *synchronizes* relation can also be established by certain CUDA APIs.

1. Completion of a task enqueued in a CUDA stream *synchronizes* with the start of the following
   task in the same stream, if any.
2. For purposes of the above, recording or waiting on a CUDA event in a stream, or causing a
   cross-stream barrier to be inserted due to `cudaStreamLegacy`, enqueues tasks in the associated
   streams even if there are no direct side effects. An event record task *synchronizes* with
   matching event wait tasks, and a barrier arrival task *synchronizes* with matching barrier wait
   tasks.
3. Start of a CUDA kernel *synchronizes* with start of all threads in the kernel. End of all threads
   in a kernel *synchronize* with end of the kernel.
4. Start of a CUDA graph *synchronizes* with start of all source nodes in the graph. Completion of
   all sink nodes in a CUDA graph *synchronizes* with completion of the graph. Completion of a graph
   node *synchronizes* with start of all nodes with a direct dependency.
5. Start of a CUDA API call to enqueue a task *synchronizes* with start of the task.
6. Completion of the last task queued to a stream, if any, *synchronizes* with return from
   `cudaStreamSynchronize`. Completion of the most recently queued matching event record task, if
   any, *synchronizes* with return from `cudaEventSynchronize`. Synchronizing a CUDA device or
   context behaves as if synchronizing all streams in the context, including ones that have been
   destroyed.
7. Returning `cudaSuccess` from an API to query a CUDA handle, such as a stream or event, behaves
   the same as return from the matching synchronization API.

In addition to establishing a *synchronizes* relation, the CUDA API synchronization mechanisms above
also participate in *proxy-preserved base causality order*.

### 8.9.5. [Causality Order](#causality-order)[](#causality-order "Permalink to this headline")

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

### 8.9.6. [Coherence Order](#coherence-order)[](#coherence-order "Permalink to this headline")

There exists a partial transitive order that relates *overlapping* write operations, determined at
runtime, called the *coherence order*1. Two *overlapping* write operations are related in
*coherence order* if they are *morally strong* or if they are related in *causality order*. Two
*overlapping* writes are unrelated in *coherence order* if they are in a *data-race*, which gives
rise to the partial nature of *coherence order*.

1 *Coherence order* cannot be observed directly since it consists entirely of write
operations. It may be observed indirectly by its use in constraining the set of candidate
writes that a read operation may read from.

### 8.9.7. [Communication Order](#communication-order)[](#communication-order "Permalink to this headline")

The *communication order* is a non-transitive order, determined at runtime, that relates write
operations to other *overlapping* memory operations.

1. A write W precedes an *overlapping* read R in *communication order* if R returns the value of any
   byte that was written by W.
2. A write W precedes a write W’ in *communication order* if W precedes W’ in *coherence order*.
3. A read R precedes an *overlapping* write W in *communication order* if, for any byte accessed by
   both R and W, R returns the value written by a write W’ that precedes W in *coherence order*.

*Communication order* captures the visibility of memory operations — when a memory operation X1
precedes a memory operation X2 in *communication order*, X1 is said to be visible to X2.