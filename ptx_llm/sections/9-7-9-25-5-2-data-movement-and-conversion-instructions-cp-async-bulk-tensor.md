###### 9.7.9.25.5.2. Data Movement and Conversion Instructions: cp.async.bulk.tensor 

`cp.async.bulk.tensor`

Initiates an asynchronous copy operation on the tensor data from one state space to another.

Syntax

```
// global -> shared::cta

cp.async.bulk.tensor.dim.dst.src{.load_mode}.completion_mechanism{.cta_group}{.level::cache_hint}

                                   [dstMem], [tensorMap, tensorCoords], [mbar]{, im2colInfo} {, cache-policy}



.dst =                  { .shared::cta }

.src =                  { .global }

.dim =                  { .1d, .2d, .3d, .4d, .5d }

.completion_mechanism = { .mbarrier::complete_tx::bytes }

.cta_group =            { .cta_group::1, .cta_group::2 }

.load_mode =            { .tile, .tile::gather4, .im2col, .im2col::w, .im2col::w::128 }

.level::cache_hint =    { .L2::cache_hint }





// global -> shared::cluster

cp.async.bulk.tensor.dim.dst.src{.load_mode}.completion_mechanism{.multicast}{.cta_group}{.level::cache_hint}

                                   [dstMem], [tensorMap, tensorCoords], [mbar]{, im2colInfo}

                                   {, ctaMask} {, cache-policy}



.dst =                  { .shared::cluster }

.src =                  { .global }

.dim =                  { .1d, .2d, .3d, .4d, .5d }

.completion_mechanism = { .mbarrier::complete_tx::bytes }

.cta_group =            { .cta_group::1, .cta_group::2 }

.load_mode =            { .tile, .tile::gather4, .im2col, .im2col::w, .im2col::w::128 }

.level::cache_hint =    { .L2::cache_hint }

.multicast =            { .multicast::cluster  }





// shared::cta -> global

cp.async.bulk.tensor.dim.dst.src{.load_mode}.completion_mechanism{.level::cache_hint}

                                   [tensorMap, tensorCoords], [srcMem] {, cache-policy}



.dst =                  { .global }

.src =                  { .shared::cta }

.dim =                  { .1d, .2d, .3d, .4d, .5d }

.completion_mechanism = { .bulk_group }

.load_mode =            { .tile, .tile::scatter4, .im2col_no_offs }

.level::cache_hint =    { .L2::cache_hint }
```

Description

`cp.async.bulk.tensor` is a non-blocking instruction which initiates an asynchronous copy
operation of tensor data from the location in `.src` state space to the location in the `.dst`
state space.

The operand `dstMem` specifies the location in the `.dst` state space into which the tensor data
has to be copied and `srcMem` specifies the location in the `.src` state space from which the
tensor data has to be copied.

When `.dst` is specified as `.shared::cta`, the address `dstMem` must be in the shared memory
of the executing CTA within the cluster, otherwise the behavior is undefined.

When `.dst` is specified as `.shared::cluster`, the address `dstMem` can be in the shared memory
of any of the CTAs within the current cluster.

The operand `tensorMap` is the generic address of the opaque tensor-map object which resides
in `.param` space or `.const` space or `.global` space. The operand `tensorMap` specifies
the properties of the tensor copy operation, as described in [Tensor-map](#tensor-tensormap).
The `tensorMap` is accessed in tensormap proxy. Refer to the *CUDA programming guide* for creating
the tensor-map objects on the host side.

The dimension of the tensor data is specified by the `.dim` modifier.

The vector operand `tensorCoords` specifies the starting coordinates in the tensor data in the
global memory from or to which the copy operation has to be performed. The individual tensor
coordinates in `tensorCoords` are of type `.s32`. The format of vector argument `tensorCoords`
is dependent on `.load_mode` specified and is as follows:

| .load\_mode | tensorCoords | Semantics |
| --- | --- | --- |
| `.tile::scatter4` | {col\_idx, row\_idx0, row\_idx1, row\_idx2, row\_idx3} | Fixed length vector of size 5. The five elements together specify the start co-ordinates of the four rows. |
| `.tile::gather4` |
| Rest all | {d0, .., dn} for n = .dim | Vector of n elements where n = .dim. The elements indicate the offset in each of the dimension. |

The modifier `.completion_mechanism` specifies the completion mechanism that is supported on the
instruction variant. The completion mechanisms that are supported for different variants are
summarized in the following table:

| .completion-mechanism | `.dst` | `.src` | Completion mechanism | |
| --- | --- | --- | --- | --- |
| Needed for completion of entire Async operation | optionally can be used for the completion of - Reading data from the source - Reading from the tensormap, if applicable |
| `.mbarrier::...` | `.shared::cta` | `.global` | mbarrier based | *Bulk async-group* based |
| `.shared::cluster` | `.global` |
| `.bulk_group` | `.global` | `.shared::cta` | *Bulk async-group* based |

The modifier `.mbarrier::complete_tx::bytes` specifies that the `cp.async.bulk.tensor` variant
uses mbarrier based completion mechanism. Upon the completion of the asynchronous copy operation, the
[complete-tx](#parallel-synchronization-and-communication-instructions-mbarrier-complete-tx-operation)
operation, with `completeCount` argument equal to amount of data copied in bytes, will be
performed on the mbarrier object specified by the operand `mbar`.

The modifier `.cta_group` can only be specified with the mbarrier based completion mechanism. The
modifier `.cta_group` is used to signal either the odd numbered CTA or the even numbered CTA among
the [CTA-Pair](#tcgen05-cta-pair). When `.cta_group::1` is specified, the mbarrier object `mbar`
that is specified must be in the shared memory of the same CTA as the shared memory destination `dstMem`.
When `.cta_group::2` is specified, the mbarrier object `mbar` can be in shared memory of either the
same CTA as the shared memory destination `dstMem` or in its [peer-CTA](#tcgen05-peer-cta). If
`.cta_group` is not specified, then it defaults to `.cta_group::1`.

The modifier `.bulk_group` specifies that the `cp.async.bulk.tensor` variant uses *bulk
async-group* based completion mechanism.

The qualifier `.load_mode` specifies how the data in the source location is copied into the
destination location. If `.load_mode` is not specified, it defaults to `.tile`.

In `.tile` mode, the multi-dimensional layout of the source tensor is preserved at the destination.
In `.tile::gather4` mode, four rows in 2-dimnesional source tensor are combined to form a single 2-dimensional
destination tensor. In `.tile::scatter4` mode, single 2-dimensional source tensor is divided into four rows
in 2-dimensional destination tensor. Details of `.tile::scatter4`/`.tile::gather4` modes are described
in [.tile::scatter4 and .tile::gather4 modes](#tensor-tiled-scatter4-gather4-modes).

In `.im2col` and `.im2col::*` modes, some dimensions of the source tensors are unrolled in a single
dimensional column at the destination. Details of the `im2col` and `.im2col::*` modes are described
in [im2col mode](#tensor-im2col-mode) and [im2col::w and im2col::w::128 modes](#tensor-im2col-w-w128-modes)
respectively. In `.im2col` and `.im2col::*` modes, the tensor has to be at least 3-dimensional. The vector
operand `im2colInfo` can be specified only when `.load_mode` is `.im2col` or `.im2col::w` or
`.im2col::w::128`. The format of the vector argument `im2colInfo` is dependent on the exact im2col mode
and is as follows:

| Exact im2col mode | im2colInfo argument | Semantics |
| --- | --- | --- |
| `.im2col` | { i2cOffW , i2cOffH , i2cOffD } for `.dim` = `.5d` | A vector of im2col offsets whose vector size is two less than number of dimensions .dim. |
| `.im2col::w` | { wHalo, wOffset } | A vector of 2 arguments containing [wHalo](#tensor-im2col-w-w128-modes-whalo) and [wOffset](#tensor-im2col-w-w128-modes-woffset) arguments. |
| `.im2col::w::128` |
| `.im2col_no_offs` | `im2colInfo` is not applicable. | `im2colInfo` is not applicable. |

Argument `wHalo` is a 16bit unsigned integer whose valid set of values differs on the load-mode and is as follows:
- Im2col::w mode : valid range is [0, 512).
- Im2col::w::128 mode : valid range is [0, 32).

Argument `wOffset` is a 16bit unsigned integer whose valid range of values is [0, 32).

The optional modifier `.multicast::cluster` allows copying of data from global memory to shared
memory of multiple CTAs in the cluster. Operand `ctaMask` specifies the destination CTAs in the
cluster such that each bit position in the 16-bit `ctaMask` operand corresponds to the `%ctaid`
of the destination CTA. The source data is multicast to the same offset as `dstMem` in the shared
memory of each destination CTA. When `.cta_group` is specified as:

* `.cta_group::1` : The mbarrier signal is also multicasted to the same offset as `mbar` in
  the shared memory of the destination CTA.
* `.cta_group::2` : The mbarrier signal is multicasted either to all the odd numbered CTAs or the
  even numbered CTAs within the corresponding [CTA-Pair](#tcgen05-cta-pair). For each destination
  CTA specified in the `ctaMask`, the mbarrier signal is sent either to the destination CTA or its
  [peer-CTA](#tcgen05-peer-cta) based on CTAs `%cluster_ctarank` parity of shared memory where
  the mbarrier object `mbar` resides.

When the optional argument `cache-policy` is specified, the qualifier `.level::cache_hint` is
required. The 64-bit operand `cache-policy` specifies the cache eviction policy that may be used
during the memory access.

`cache-policy` is a hint to the cache subsystem and may not always be respected. It is treated as
a performance hint only, and does not change the memory consistency behavior of the program.

The copy operation in `cp.async.bulk.tensor` is treated as a weak memory operation and the
[complete-tx](#parallel-synchronization-and-communication-instructions-mbarrier-complete-tx-operation)
operation on the mbarrier has `.release` semantics at the `.cluster` scope as described in the
[Memory Consistency Model](#memory-consistency-model).

Notes

`.multicast::cluster` qualifier is optimized for target architecture `sm_90a`/`sm_100f`/`sm_100a`/
`sm_103f`/`sm_103a`/`sm_110f`/`sm_110a` and may have substantially reduced performance on other
targets and hence `.multicast::cluster` is advised to be used with `.target` `sm_90a`/`sm_100f`/
`sm_100a`/`sm_103f`/`sm_103a`/`sm_110f`/`sm_110a`.

PTX ISA Notes

Introduced in PTX ISA version 8.0.

Support for `.shared::cta` as destination state space is introduced in PTX ISA version 8.6.

Support for qualifiers `.tile::gather4` and `.tile::scatter4` introduced in PTX ISA version 8.6.

Support for qualifiers `.im2col::w` and `.im2col::w::128` introduced in PTX ISA version 8.6.

Support for qualifier `.cta_group` introduced in PTX ISA version 8.6.

Target ISA Notes

Requires `sm_90` or higher.

`.multicast::cluster` qualifier advised to be used with `.target` `sm_90a` or `sm_100f` or
`sm_100a` or `sm_103f` or `sm_103a` or `sm_110f` or `sm_110a`.

Qualifiers `.tile::gather4` and `.im2col::w` require:

* `sm_100a` when destination state space is `.shared::cluster` and is supported on `sm_100f` from PTX ISA version 8.8.
* `sm_100` or higher when destination state space is `.shared::cta`.

Qualifier `.tile::scatter4` is supported on following architectures:

* `sm_100a`
* `sm_101a` (Renamed to `sm_110a` from PTX ISA version 9.0)
* And is supported on following family-specific architectures from PTX ISA version 8.8:

  + `sm_100f` or higher in the same family
  + `sm_101f` or higher in the same family (Renamed to `sm_110f` from PTX ISA version 9.0)
* `sm_110f` or higher in the same family

Qualifier `.im2col::w::128` is supported on following architectures:

* `sm_100a`
* `sm_101a` (Renamed to `sm_110a` from PTX ISA version 9.0)
* And is supported on following family-specific architectures from PTX ISA version 8.8:

  + `sm_100f` or higher in the same family
  + `sm_101f` or higher in the same family (Renamed to `sm_110f` from PTX ISA version 9.0)
* `sm_110f` or higher in the same family

Qualifier `.cta_group` is supported on following architectures:

* `sm_100a`
* `sm_101a` (Renamed to `sm_110a` from PTX ISA version 9.0)
* And is supported on following family-specific architectures from PTX ISA version 8.8:

  + `sm_100f` or higher in the same family
  + `sm_101f` or higher in the same family (Renamed to `sm_110f` from PTX ISA version 9.0)
* `sm_110f` or higher in the same family

Examples

```
.reg .b16 ctaMask;

.reg .u16 i2cOffW, i2cOffH, i2cOffD;

.reg .b64 l2CachePolicy;



cp.async.bulk.tensor.1d.shared::cta.global.mbarrier::complete_tx::bytes.tile  [sMem0], [tensorMap0, {tc0}], [mbar0];



@p cp.async.bulk.tensor.5d.shared::cta.global.im2col.mbarrier::complete_tx::bytes

                     [sMem2], [tensorMap2, {tc0, tc1, tc2, tc3, tc4}], [mbar2], {i2cOffW, i2cOffH, i2cOffD};



cp.async.bulk.tensor.1d.shared::cluster.global.mbarrier::complete_tx::bytes.tile  [sMem0], [tensorMap0, {tc0}], [mbar0];



@p cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster

                     [sMem1], [tensorMap1, {tc0, tc1}], [mbar2], ctaMask;



@p cp.async.bulk.tensor.5d.shared::cluster.global.im2col.mbarrier::complete_tx::bytes

                     [sMem2], [tensorMap2, {tc0, tc1, tc2, tc3, tc4}], [mbar2], {i2cOffW, i2cOffH, i2cOffD};



@p cp.async.bulk.tensor.3d.im2col.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint

                     [sMem3], [tensorMap3, {tc0, tc1, tc2}], [mbar3], {i2cOffW}, policy;



@p cp.async.bulk.tensor.1d.global.shared::cta.bulk_group  [tensorMap3, {tc0}], [sMem3];



cp.async.bulk.tensor.2d.tile::gather4.shared::cluster.global.mbarrier::complete_tx::bytes

                     [sMem5], [tensorMap6, {x0, y0, y1, y2, y3}], [mbar5];



cp.async.bulk.tensor.3d.im2col::w.shared::cluster.global.mbarrier::complete_tx::bytes

                     [sMem4], [tensorMap5, {t0, t1, t2}], [mbar4], {im2colwHalo, im2colOff};



cp.async.bulk.tensor.1d.shared::cluster.global.tile.cta_group::2

                     [sMem6], [tensorMap7, {tc0}], [peerMbar];
```