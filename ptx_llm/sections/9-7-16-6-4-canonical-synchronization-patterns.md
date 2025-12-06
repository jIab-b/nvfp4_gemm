##### 9.7.16.6.4. Canonical synchronization patterns 

Using the above rules, the following are the five canonical synchronization patterns:

###### 9.7.16.6.4.1. [Pipelined instructions, same thread](#tcgen05-memory-consistency-model-canonical-sync-patterns-pipelined-same-thread)[](#tcgen05-memory-consistency-model-canonical-sync-patterns-pipelined-same-thread "Permalink to this headline")

In this pattern, no explicit ordering mechanism is needed and the ordering guarantee is
provided by the pipelined instruction pairing.

Example:

```
tcgen05.mma

tcgen05.mma (same shape and accumulator)
```

The two instructions will be executed in program order.

###### 9.7.16.6.4.2. [Non-pipelined instructions, same thread](#tcgen05-memory-consistency-model-canonical-sync-patterns-non-pipelined-same-thread)[](#tcgen05-memory-consistency-model-canonical-sync-patterns-non-pipelined-same-thread "Permalink to this headline")

In this pattern, explicit waiting mechanisms are used to wait for the completion of the
asynchronous `tcgen05` operations.

Example 1:

```
tcgen05.st

tcgen05.wait::st

tcgen05.ld
```

`tcgen05.wait::st` is used to wait for the completion of the prior asynchronous
instruction `tcgen05.st`.

Example 2:

```
tcgen05.mma [d], ...

tcgen05.commit.mbarrier::arrive::one

mbarrier.try_wait.relaxed.cluster (loop until successful)

tcgen05.fence::after_thread_sync

tcgen05.ld [d], ...
```

For the completion of the asynchronous `tcgen05.mma`, `tcgen05.commit` is used.

As `tcgen05.ld` is an asynchronous operation, the instruction `tcgen05.fence::after_thread_sync`
is needed.

No explicit `tcgen05.fence::before_thread_sync` is needed as this is implicitly performed by
`tcgen05.commit`. The combination of `tcgen05.mma` and `tcgen05.commit` forms a
conceptual asynchronous pipeline and establishes execution ordering.

```
tcgen05.mma [d], ...

tcgen05.fence::before_thread_sync

mbarrier::arrive
```

###### 9.7.16.6.4.3. [Pipelined instructions, different thread](#tcgen05-memory-consistency-model-canonical-sync-patterns-pipelined-diff-thread)[](#tcgen05-memory-consistency-model-canonical-sync-patterns-pipelined-diff-thread "Permalink to this headline")

In this pattern, no explicit waiting mechanism is needed but proper synchronization between threads is needed.

Example:

| Thread 0 | Thread 1 |
| --- | --- |
| ``` tcgen05.cp  tcgen05.fence::before_thread_sync  mbarrier.arrive.relaxed.cluster ``` |  |
|  | ``` mbarrier.try_wait.relaxed.cluster // loop till success  tcgen05.fence::after_thread_sync  tcgen05.mma ``` |

###### 9.7.16.6.4.4. [Non-pipelined instructions, different thread](#tcgen05-memory-consistency-model-canonical-sync-patterns-non-pipelined-diff-thread)[](#tcgen05-memory-consistency-model-canonical-sync-patterns-non-pipelined-diff-thread "Permalink to this headline")

In this pattern, the producer threads that issue the asynchronous `tcgen05` instructions
must explicitly wait for the instructions’ completion before synchronizing with the consumer threads.

Example 1:

| Thread 0 | Thread 1 |
| --- | --- |
| ``` tcgen05.ld  tcgen05.wait::ld  tcgen05.fence::before_thread_sync  mbarrier.arrive.relaxed.cluster ``` |  |
|  | ``` mbarrier.try_wait.relaxed.cluster // loop till success  tcgen05.fence::after_thread_sync  tcgen05.mma ``` |

Example 1:

| Thread 0 | Thread 1 |
| --- | --- |
| ``` tcgen05.mma  tcgen05.commit.mbarrier::arrive::one [mbar] ``` |  |
|  | ``` mbarrier.try_wait.relaxed.cluster [mbar] // loop till success  tcgen05.fence::after_thread_sync  tcgen05.ld ``` |

The synchronization mechanisms can also be composed with each other. For example:

| Thread 0 | Thread 1 |
| --- | --- |
| ``` tcgen05.mma  tcgen05.commit.mbarrier::arrive::one [bar1]  mbarrier.try_wait.relaxed.cluster [bar1] // loop  ...  tcgen05.fence::after_thread_sync  ...// completion is guaranteed  tcgen05.fence::before_thread_sync  mbarrier.arrive.relaxed.cluster [bar2] // loop  ... ``` |  |
|  | ``` mbarrier.try_wait.relaxed.cluster [bar2] // loop  ...  tcgen05.fence::after_thread_sync  tcgen05.ld ``` |

###### 9.7.16.6.4.5. [Register dependencies, same thread](#tcgen05-memory-consistency-model-canonical-sync-patterns-reg-dependency-same-thread)[](#tcgen05-memory-consistency-model-canonical-sync-patterns-reg-dependency-same-thread "Permalink to this headline")

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