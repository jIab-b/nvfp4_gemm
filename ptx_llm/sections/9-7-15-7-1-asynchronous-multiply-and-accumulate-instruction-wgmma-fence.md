##### 9.7.15.7.1. Asynchronous Multiply-and-Accumulate Instruction: wgmma.fence 

`wgmma.fence`

Enforce an ordering of register accesses between `wgmma.mma_async` and other operations.

Syntax

```
wgmma.fence.sync.aligned;
```

Description

`wgmma.fence` instruction establishes an ordering between prior accesses to any warpgroup
registers and subsequent accesses to the same registers by a `wgmma.mma_async` instruction. Only
the accumulator register and the input registers containing the fragments of matrix A require this
ordering.

The `wgmma.fence` instruction must be issued by all warps of the warpgroup at the following
locations:

* Before the first `wgmma.mma_async` operation in a warpgroup.
* Between a register access by a thread in the warpgroup and any `wgmma.mma_async` instruction
  that accesses the same registers, either as accumulator or input register containing fragments of
  matrix A, except when these are accumulator register accesses across multiple `wgmma.mma_async`
  instructions of the same shape. In the latter case, an ordering guarantee is provided by default.

Otherwise, the behavior is undefined.

An async proxy fence must be used to establish an ordering between prior writes to shared memory
matrices and subsequent reads of the same matrices in a `wgmma.mma_async` instruction.

The mandatory `.sync` qualifier indicates that `wgmma.fence` instruction causes the executing
thread to wait until all threads in the warp execute the same `wgmma.fence` instruction before
resuming execution.

The mandatory `.aligned` qualifier indicates that all threads in the warpgroup must execute the
same `wgmma.fence` instruction. In conditionally executed code, an `wgmma.fence` instruction
should only be used if it is known that all threads in the warpgroup evaluate the condition
identically, otherwise the behavior is undefined.

PTX ISA Notes

Introduced in PTX ISA version 8.0.

Target ISA Notes

Requires `sm_90a`.

Examples

```
// Example 1, first use example:

wgmma.fence.sync.aligned;    // Establishes an ordering w.r.t. prior accesses to the registers s32d<0-3>

wgmma.mma_async.sync.aligned.m64n8k32.s32.u8.u8  {s32d0, s32d1, s32d2, s32d3},

                                                  descA, descB, scaleD;

wgmma.commit_group.sync.aligned;

wgmma.wait_group.sync.aligned 0;



// Example 2, use-case with the input value updated in between:

wgmma.fence.sync.aligned;

wgmma.mma_async.sync.aligned.m64n8k32.s32.u8.u8  {s32d0, s32d1, s32d2, s32d3},

                                                  descA, descB, scaleD;

...

mov.b32 s32d0, new_val;

wgmma.fence.sync.aligned;

wgmma.mma_async.sync.aligned.m64n8k32.s32.u8.u8  {s32d4, s32d5, s32d6, s32d7},

                                                 {s32d0, s32d1, s32d2, s32d3},

                                                  descB, scaleD;

wgmma.commit_group.sync.aligned;

wgmma.wait_group.sync.aligned 0;
```