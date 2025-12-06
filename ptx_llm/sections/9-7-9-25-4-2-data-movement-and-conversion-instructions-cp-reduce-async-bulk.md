###### 9.7.9.25.4.2. Data Movement and Conversion Instructions: cp.reduce.async.bulk 

`cp.reduce.async.bulk`

Initiates an asynchronous reduction operation.

Syntax

```
cp.reduce.async.bulk.dst.src.completion_mechanism.redOp.type

              [dstMem], [srcMem], size, [mbar]



.dst =                  { .shared::cluster }

.src =                  { .shared::cta }

.completion_mechanism = { .mbarrier::complete_tx::bytes }

.redOp=                 { .and, .or, .xor,

                          .add, .inc, .dec,

                          .min, .max }

.type =                 { .b32, .u32, .s32, .b64, .u64 }





cp.reduce.async.bulk.dst.src.completion_mechanism{.level::cache_hint}.redOp.type

               [dstMem], [srcMem], size{, cache-policy}



.dst =                  { .global      }

.src =                  { .shared::cta }

.completion_mechanism = { .bulk_group }

.level::cache_hint    = { .L2::cache_hint }

.redOp=                 { .and, .or, .xor,

                          .add, .inc, .dec,

                          .min, .max }

.type =                 { .f16, .bf16, .b32, .u32, .s32, .b64, .u64, .s64, .f32, .f64 }





cp.reduce.async.bulk.dst.src.completion_mechanism{.level::cache_hint}.add.noftz.type

               [dstMem], [srcMem], size{, cache-policy}

.dst  =                 { .global }

.src  =                 { .shared::cta }

.completion_mechanism = { .bulk_group }

.type =                 { .f16, .bf16 }
```

Description

`cp.reduce.async.bulk` is a non-blocking instruction which initiates an asynchronous reduction
operation on an array of memory locations specified by the destination address operand `dstMem`
with the source array whose location is specified by the source address operand `srcMem`. The size
of the source and the destination array must be the same and is specified by the operand `size`.

Each data element in the destination array is reduced inline with the corresponding data element in
the source array with the reduction operation specified by the modifier `.redOp`. The type of each
data element in the source and the destination array is specified by the modifier `.type`.

The source address operand `srcMem` is located in the state space specified by `.src` and the
destination address operand `dstMem` is located in the state specified by the `.dst`.

The 32-bit operand `size` specifies the amount of memory to be copied from the source location and
used in the reduction operation, in terms of number of bytes. `size` must be a multiple of 16. If
the value is not a multiple of 16, then the behavior is undefined. The memory range `[dstMem,
dstMem + size - 1]` must not overflow the destination memory space and the memory range `[srcMem,
srcMem + size - 1]` must not overflow the source memory space. Otherwise, the behavior is
undefined. The addresses `dstMem` and `srcMem` must be aligned to 16 bytes.

The operations supported by `.redOp` are classified as follows:

* The bit-size operations are `.and`, `.or`, and `.xor`.
* The integer operations are `.add`, `.inc`, `.dec`, `.min`, and `.max`. The `.inc` and
  `.dec` operations return a result in the range `[0..x]` where `x` is the value at the source
  state space.
* The floating point operation `.add` rounds to the nearest even. The current implementation of
  `cp.reduce.async.bulk.add.f32` flushes subnormal inputs and results to sign-preserving zero. The
  `cp.reduce.async.bulk.add.f16` and `cp.reduce.async.bulk.add.bf16` operations require
  `.noftz` qualifier. It preserves input and result subnormals, and does not flush them to zero.

The following table describes the valid combinations of `.redOp` and element type:

| `.dst` | `.redOp` | Element type |
| --- | --- | --- |
| `.shared::cluster` | `.add` | `.u32`, `.s32`, `.u64` |
| `.min`, `.max` | `.u32`, `.s32` |
| `.inc`, `.dec` | `.u32` |
| `.and`, `.or`, `.xor` | `.b32` |
| `.global` | `.add` | `.u32`, `.s32`, `.u64`, `.f32`, `.f64`, `.f16`, `.bf16` |
| `.min`, `.max` | `.u32`, `.s32`, `.u64`, `.s64`, `.f16`, `.bf16` |
| `.inc`, `.dec` | `.u32` |
| `.and`, `.or`, `.xor` | `.b32`, `.b64` |

The modifier `.completion_mechanism` specifies the completion mechanism that is supported on the
instruction variant. The completion mechanisms that are supported for different variants are
summarized in the following table:

| .completion-mechanism | `.dst` | `.src` | Completion mechanism | |
| --- | --- | --- | --- | --- |
| Needed for completion of entire Async operation | optionally can be used for the completion of - Reading data from the source - Reading from the tensormap, if applicable |
| `.mbarrier::...` | `.shared::cluster` | `.global` | mbarrier based | *Bulk async-group* based |
| `.shared::cluster` | `.shared::cta` |
| `.bulk_group` | `.global` | `.shared::cta` | *Bulk async-group* based |

The modifier `.mbarrier::complete_tx::bytes` specifies that the `cp.reduce.async.bulk` variant
uses mbarrier based completion mechanism. The [complete-tx](#parallel-synchronization-and-communication-instructions-mbarrier-complete-tx-operation)
operation, with `completeCount` argument equal to amount of data copied in bytes, will be
performed on the mbarrier object specified by the operand `mbar`.

The modifier `.bulk_group` specifies that the `cp.reduce.async.bulk` variant uses *bulk
async-group* based completion mechanism.

When the optional argument `cache-policy` is specified, the qualifier `.level::cache_hint` is
required. The 64-bit operand `cache-policy` specifies the cache eviction policy that may be used
during the memory access.

`cache-policy` is a hint to the cache subsystem and may not always be respected. It is treated as
a performance hint only, and does not change the memory consistency behavior of the program. The
qualifier `.level::cache_hint` is only supported when at least one of the `.src` or `.dst`
statespaces is `.global` state space.

Each reduction operation performed by the `cp.reduce.async.bulk` has individually `.relaxed.gpu`
memory ordering semantics. The load operations in `cp.reduce.async.bulk` are treated as weak
memory operation and the [complete-tx](#parallel-synchronization-and-communication-instructions-mbarrier-complete-tx-operation)
operation on the mbarrier has `.release` semantics at the `.cluster` scope as described in the
[Memory Consistency Model](#memory-consistency-model).

PTX ISA Notes

Introduced in PTX ISA version 8.0.

Target ISA Notes

Requires `sm_90` or higher.

Examples

```
cp.reduce.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes.add.u64

                                                                  [dstMem], [srcMem], size, [mbar];



cp.reduce.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes.min.s32

                                                                  [dstMem], [srcMem], size, [mbar];



cp.reduce.async.bulk.global.shared::cta.bulk_group.min.f16 [dstMem], [srcMem], size;



cp.reduce.async.bulk.global.shared::cta.bulk_group.L2::cache_hint.xor.s32 [dstMem], [srcMem], size, policy;



cp.reduce.async.bulk.global.shared::cta.bulk_group.add.noftz.f16 [dstMem], [srcMem], size;
```