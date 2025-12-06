##### 9.7.15.7.3. Asynchronous Multiply-and-Accumulate Instruction: wgmma.wait_group 

`wgmma.wait_group`

Signal the completion of a preceding warpgroup operation.

Syntax

```
wgmma.wait_group.sync.aligned N;
```

Description

`wgmma.wait_group` instruction will cause the executing thread to wait until only N or fewer of
the most recent wgmma-groups are pending and all the prior wgmma-groups committed by the executing
threads are complete. For example, when N is 0, the executing thread waits on all the prior
wgmma-groups to complete. Operand N is an integer constant.

Accessing the accumulator register or the input register containing the fragments of matrix A of a
`wgmma.mma_async` instruction without first performing a `wgmma.wait_group` instruction that
waits on a *wgmma-group* including that `wgmma.mma_async` instruction is undefined behavior.

The mandatory `.sync` qualifier indicates that `wgmma.wait_group` instruction causes the
executing thread to wait until all threads in the warp execute the same `wgmma.wait_group`
instruction before resuming execution.

The mandatory `.aligned` qualifier indicates that all threads in the warpgroup must execute the
same `wgmma.wait_group` instruction. In conditionally executed code, an `wgmma.wait_group`
instruction should only be used if it is known that all threads in the warpgroup evaluate the
condition identically, otherwise the behavior is undefined.

PTX ISA Notes

Introduced in PTX ISA version 8.0.

Target ISA Notes

Requires `sm_90a`.

Examples

```
wgmma.fence.sync.aligned;



wgmma.mma_async.sync.aligned.m64n8k32.s32.u8.u8  {s32d0, s32d1, s32d2, s32d3},

                                                  descA, descB, scaleD;

wgmma.commit_group.sync.aligned;



wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 {f32d0, f32d1, f32d2, f32d3},

                                                  {f16a0, f16a1, f16a2, f16a3},

                                                   descB, 1, -1, -1, 1;

wgmma.commit_group.sync.aligned;



wgmma.wait_group.sync.aligned 0;
```