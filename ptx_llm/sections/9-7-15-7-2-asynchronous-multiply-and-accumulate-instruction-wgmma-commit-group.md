##### 9.7.15.7.2. Asynchronous Multiply-and-Accumulate Instruction: wgmma.commit_group 

`wgmma.commit_group`

Commits all prior uncommitted `wgmma.mma_async` operations into a *wgmma-group*.

Syntax

```
wgmma.commit_group.sync.aligned;
```

Description

`wgmma.commit_group` instruction creates a new wgmma-group per warpgroup and batches all prior
`wgmma.mma_async` instructions initiated by the executing warp but not committed to any
wgmma-group into the new wgmma-group. If there are no uncommitted `wgmma.mma_async` instructions
then `wgmma.commit_group` results in an empty wgmma-group.

An executing thread can wait for the completion of all `wgmma.mma_async` operations in a
wgmma-group by using `wgmma.wait_group`.

The mandatory `.sync` qualifier indicates that `wgmma.commit_group` instruction causes the
executing thread to wait until all threads in the warp execute the same `wgmma.commit_group`
instruction before resuming execution.

The mandatory `.aligned` qualifier indicates that all threads in the warpgroup must execute the
same `wgmma.commit_group` instruction. In conditionally executed code, an `wgmma.commit_group`
instruction should only be used if it is known that all threads in the warpgroup evaluate the
condition identically, otherwise the behavior is undefined.

PTX ISA Notes

Introduced in PTX ISA version 8.0.

Target ISA Notes

Requires `sm_90a`.

Examples

```
wgmma.commit_group.sync.aligned;
```