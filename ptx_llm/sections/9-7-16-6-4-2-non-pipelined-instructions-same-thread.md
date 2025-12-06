###### 9.7.16.6.4.2. Non-pipelined instructions, same thread 

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