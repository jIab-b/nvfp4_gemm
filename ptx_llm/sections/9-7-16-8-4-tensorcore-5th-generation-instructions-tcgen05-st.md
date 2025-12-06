##### 9.7.16.8.4. Tensorcore 5th Generation Instructions: tcgen05.st 

`tcgen05.st`

Asynchronous collective store to tensor memory from registers.

Syntax

```
tcgen05.st.sync.aligned.shape1.num{.unpack}.b32    [taddr], r;



tcgen05.st.sync.aligned.shape2.num{.unpack}.b32    [taddr], immHalfSplitoff, r;



.shape1 = { .16x64b, .16x128b, .16x256b, .32x32b }

.shape2 = { .16x32bx2 }

.num    = { .x1, .x2, .x4, .x8, .x16, .x32, .x64, .x128 }

.unpack = { .unpack::16b }
```

Description

Instruction `tcgen05.st` asynchronously stores data from the source register `r` into
the [Tensor Memory](#tensor-memory) at the location specified by the 32-bit address operand `taddr`,
collectively across all threads of the warps.

All the threads in the warp must specify the same value of `taddr`, which must be the base
address of the collective store operation. Otherwise, the behavior is undefined.

The `.shape` qualifier and the `.num` qualifier together determines the total dimension
of the data which is stored to the Tensor Memory. The `.shape` qualifier indicates the base
dimension of data to be accessed as described in the
[Data Movement Shape](#tcgen05-data-movement-shape). The `.num`
qualifier indicates the repeat factor on the base dimension resulting in the total dimension of
the data that is accessed.

The shape `.16x32bx2` performs two accesses into Tensor Memory of the shape `.16x32b`.
The base address of the first access is specified by `taddr` and the base address of the
second access is specified by `taddr+immHalfSplitoff`, where `immHalfSplitoff` is an
immediate argument.

The source operand `r` is a brace-enclosed vector expression consisting of one or more 32-bit
registers as per the value of `.shape` and `.num`. The size of the vector for various
combinations of `.num` and `.shape` is shown in [Table 48](#tcgen05-num-shapes-st).

Table 48 Various-combinations of .num and .shape[](#tcgen05-num-shapes-st "Permalink to this table")






| .num | .shape | | |
| --- | --- | --- | --- |
| .16x32bx2 / .16x64b / .32x32b | .16x128b | .16x256b |
| `.x1` | 1 | 2 | 4 |
| `.x2` | 2 | 4 | 8 |
| `.x4` | 4 | 8 | 16 |
| `.x8` | 8 | 16 | 32 |
| `.x16` | 16 | 32 | 64 |
| `.x32` | 32 | 64 | 128 |
| `.x64` | 64 | 128 | NA |
| `.x128` | 128 | NA | NA |

The optional qualifier `.unpack::16b` can be used to unpack a 32-bit element in the
register into two 16-bit elements and store them in adjacent columns as shown in the
section [Packing and Unpacking](#tcgen05-tensor-memory-ld-st-packing-unpacking).

The mandatory `.sync` qualifier indicates that `tcgen05.st` causes the executing
thread to wait until all threads in the warp execute the same `tcgen05.st` instruction
before resuming execution.

The mandatory `.aligned` qualifier indicates that all threads in the warp must execute
the same `tcgen05.st` instruction. In conditionally executed code, a `tcgen05.st`
instruction should only be used if it is known that all threads in the warp evaluate
the condition identically, otherwise behavior is undefined.

The behavior of `tcgen05.st` is undefined if all threads do not use the same values of
`taddr`, or if any thread in the warp has exited.

The instruction `tcgen05.st` is performed asynchronously and more details are specified
in the section [Memory Consistency Model for 5th generation of TensorCore operations](#tcgen05-memory-consistency-model).

PTX ISA Notes

Introduced in PTX ISA version 8.6.

Target ISA Notes

Supported on following architectures:

* `sm_100a`
* `sm_101a` (Renamed to `sm_110a` from PTX ISA version 9.0)
* And is supported on following family-specific architectures from PTX ISA version 8.8:

  + `sm_100f` or higher in the same family
  + `sm_101f` or higher in the same family (Renamed to `sm_110f` from PTX ISA version 9.0)
* `sm_110f` or higher in the same family

Examples

```
tcgen05.st.sync.aligned.16x64b.x4.b32               [taddr0], {r0,  r1,  r2,  r3};



tcgen05.st.sync.aligned.16x128b.x1.unpack::16b.b32  [taddr1], {r0,  r1};
```