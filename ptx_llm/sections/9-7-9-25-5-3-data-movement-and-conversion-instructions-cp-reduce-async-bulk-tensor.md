###### 9.7.9.25.5.3. Data Movement and Conversion Instructions: cp.reduce.async.bulk.tensor 

`cp.reduce.async.bulk.tensor`

Initiates an asynchronous reduction operation on the tensor data.

Syntax

```
// shared::cta -> global:

cp.reduce.async.bulk.tensor.dim.dst.src.redOp{.load_mode}.completion_mechanism{.level::cache_hint}

                                          [tensorMap, tensorCoords], [srcMem] {,cache-policy}



.dst =                  { .global }

.src =                  { .shared::cta }

.dim =                  { .1d, .2d, .3d, .4d, .5d }

.completion_mechanism = { .bulk_group }

.load_mode =            { .tile, .im2col_no_offs }

.redOp =                { .add, .min, .max, .inc, .dec, .and, .or, .xor}
```

Description

`cp.reduce.async.bulk.tensor` is a non-blocking instruction which initiates an asynchronous
reduction operation of tensor data in the `.dst` state space with tensor data in the `.src`
state space.

The operand `srcMem` specifies the location of the tensor data in the `.src` state space using
which the reduction operation has to be performed.

The operand `tensorMap` is the generic address of the opaque tensor-map object which resides
in `.param` space or `.const` space or `.global` space. The operand `tensorMap` specifies
the properties of the tensor copy operation, as described in [Tensor-map](#tensor-tensormap).
The `tensorMap` is accessed in tensormap proxy. Refer to the *CUDA programming guide* for creating
the tensor-map objects on the host side.

Each element of the tensor data in the `.dst` state space is reduced inline with the corresponding
element from the tensor data in the `.src` state space. The modifier `.redOp` specifies the
reduction operation used for the inline reduction. The type of each tensor data element in the
source and the destination tensor is specified in [Tensor-map](#tensor-tensormap).

The dimension of the tensor is specified by the `.dim` modifier.

The vector operand `tensorCoords` specifies the starting coordinates of the tensor data in the
global memory on which the reduce operation is to be performed. The number of tensor coordinates in
the vector argument `tensorCoords` should be equal to the dimension specified by the modifier
`.dim`. The individual tensor coordinates are of the type `.s32`.

The following table describes the valid combinations of `.redOp` and element type:

| `.redOp` | Element type |
| --- | --- |
| `.add` | `.u32`, `.s32`, `.u64`, `.f32`, `.f16`, `.bf16` |
| `.min`, `.max` | `.u32`, `.s32`, `.u64`, `.s64`, `.f16`, `.bf16` |
| `.inc`, `.dec` | `.u32` |
| `.and`, `.or`, `.xor` | `.b32`, `.b64` |

The modifier `.completion_mechanism` specifies the completion mechanism that is supported on the
instruction variant. Value `.bulk_group` of the modifier `.completion_mechanism` specifies that
`cp.reduce.async.bulk.tensor` instruction uses *bulk async-group* based completion mechanism.

The qualifier `.load_mode` specifies how the data in the source location is copied into the
destination location. If `.load_mode` is not specified, it defaults to `.tile`. In `.tile`
mode, the multi-dimensional layout of the source tensor is preserved at the destination. In
`.im2col_no_offs` mode, some dimensions of the source tensors are unrolled in a single dimensional
column at the destination. Details of the `im2col` mode are described in
[im2col mode](#tensor-im2col-mode). In `.im2col` mode, the tensor has to be at least
3-dimensional.

When the optional argument `cache-policy` is specified, the qualifier `.level::cache_hint` is
required. The 64-bit operand `cache-policy` specifies the cache eviction policy that may be used
during the memory access.

`cache-policy` is a hint to the cache subsystem and may not always be respected. It is treated as
a performance hint only, and does not change the memory consistency behavior of the program. The
qualifier `.level::cache_hint` is only supported when at least one of the `.src` or `.dst`
statespaces is `.global` state space.

Each reduction operation performed by `cp.reduce.async.bulk.tensor` has individually
`.relaxed.gpu` memory ordering semantics. The load operations in `cp.reduce.async.bulk.tensor`
are treated as weak memory operations and the [complete-tx](#parallel-synchronization-and-communication-instructions-mbarrier-complete-tx-operation)
operation on the mbarrier has `.release` semantics at the `.cluster` scope as described in the
[Memory Consistency Model](#memory-consistency-model).

PTX ISA Notes

Introduced in PTX ISA version 8.0.

Target ISA Notes

Requires `sm_90` or higher.

Examples

```
cp.reduce.async.bulk.tensor.1d.global.shared::cta.add.tile.bulk_group

                                             [tensorMap0, {tc0}], [sMem0];



cp.reduce.async.bulk.tensor.2d.global.shared::cta.and.bulk_group.L2::cache_hint

                                             [tensorMap1, {tc0, tc1}], [sMem1] , policy;



cp.reduce.async.bulk.tensor.3d.global.shared::cta.xor.im2col.bulk_group

                                             [tensorMap2, {tc0, tc1, tc2}], [sMem2]
```