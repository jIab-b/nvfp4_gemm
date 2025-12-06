###### 9.7.9.25.5.4. Data Movement and Conversion Instructions: cp.async.bulk.prefetch.tensor 

`cp.async.bulk.prefetch.tensor`

Provides a hint to the system to initiate the asynchronous prefetch of tensor data to the cache.

Syntax

```
// global -> shared::cluster:

cp.async.bulk.prefetch.tensor.dim.L2.src{.load_mode}{.level::cache_hint} [tensorMap, tensorCoords]

                                                             {, im2colInfo } {, cache-policy}



.src =                { .global }

.dim =                { .1d, .2d, .3d, .4d, .5d }

.load_mode =          { .tile, .tile::gather4, .im2col, .im2col::w, .im2col::w::128 }

.level::cache_hint =  { .L2::cache_hint }
```

Description

`cp.async.bulk.prefetch.tensor` is a non-blocking instruction which may initiate an asynchronous
prefetch of tensor data from the location in `.src` statespace to the L2 cache.

The operand `tensorMap` is the generic address of the opaque tensor-map object which resides
in `.param` space or `.const` space or `.global` space. The operand `tensorMap` specifies
the properties of the tensor copy operation, as described in [Tensor-map](#tensor-tensormap).
The `tensorMap` is accessed in tensormap proxy. Refer to the *CUDA programming guide* for creating
the tensor-map objects on the host side.

The dimension of the tensor data is specified by the `.dim` modifier.

The vector operand `tensorCoords` specifies the starting coordinates in the tensor data in the
global memory from which the copy operation has to be performed. The individual tensor
coordinates in `tensorCoords` are of type `.s32`. The format of vector argument `tensorCoords`
is dependent on `.load_mode` specified and is as follows:

| .load\_mode | tensorCoords | Semantics |
| --- | --- | --- |
| `.tile::gather4` | {col\_idx, row\_idx0, row\_idx1, row\_idx2, row\_idx3} | Fixed length vector of size 5. The five elements together specify the start co-ordinates of the four rows. |
| Rest all | {d0, .., dn} for n = .dim | Vector of n elements where n = .dim. The elements indicate the offset in each of the dimension. |

The qualifier `.load_mode` specifies how the data in the source location is copied into the
destination location. If `.load_mode` is not specified, it defaults to `.tile`.

In `.tile` mode, the multi-dimensional layout of the source tensor is preserved at the destination.
In `.tile::gather4` mode, four rows in the 2-dimnesional source tensor are fetched to L2 cache.
Details of `.tile::gather4` modes are described
in [.tile::scatter4 and .tile::gather4 modes](#tensor-tiled-scatter4-gather4-modes).

In `.im2col` and `.im2col::*` modes, some dimensions of the source tensors are unrolled in a single
dimensional column at the destination. Details of the `im2col` and `.im2col::*` modes are described in
[im2col mode](#tensor-im2col-mode) and [im2col::w and im2col::w::128 modes](#tensor-im2col-w-w128-modes)
respectively. In `.im2col` and `.im2col::*` modes, the tensor has to be at least 3-dimensional. The vector
operand `im2colInfo` can be specified only when `.load_mode` is `.im2col` or `.im2col::w` or
`.im2col::w::128`. The format of the vector argument `im2colInfo` is dependent on the exact im2col mode
and is as follows:

| Exact im2col mode | im2colInfo argument | Semantics |
| --- | --- | --- |
| `.im2col` | { i2cOffW , i2cOffH , i2cOffD } for `.dim` = `.5d` | A vector of im2col offsets whose vector size is two less than number of dimensions .dim. |
| `.im2col::w` | { wHalo, wOffset } | A vector of 2 arguments containing [wHalo](#tensor-im2col-w-w128-modes-whalo) and [wOffset](#tensor-im2col-w-w128-modes-woffset) arguments. |
| `.im2col::w::128` |

When the optional argument `cache-policy` is specified, the qualifier `.level::cache_hint` is
required. The 64-bit operand `cache-policy` specifies the cache eviction policy that may be used
during the memory access.

`cache-policy` is a hint to the cache subsystem and may not always be respected. It is treated as
a performance hint only, and does not change the memory consistency behavior of the program.

`cp.async.bulk.prefetch.tensor` is treated as a weak memory operation in the
[Memory Consistency Model](#memory-consistency-model).

PTX ISA Notes

Introduced in PTX ISA version 8.0.

Support for qualifier `.tile::gather4` introduced in PTX ISA version 8.6.

Support for qualifiers `.im2col::w` and `.im2col::w::128` introduced in PTX ISA version 8.6.

Target ISA Notes

Requires `sm_90` or higher.

Qualifier `.tile::gather4` is supported on following architectures:

* `sm_100a`
* `sm_101a` (Renamed to `sm_110a` from PTX ISA version 9.0)
* And is supported on following family-specific architectures from PTX ISA version 8.8:

  + `sm_100f` or higher in the same family
  + `sm_101f` or higher in the same family (Renamed to `sm_110f` from PTX ISA version 9.0)
* `sm_110f` or higher in the same family

Qualifiers `.im2col::w` and `.im2col::w::128` are supported on following architectures:

* `sm_100a`
* `sm_101a` (Renamed to `sm_110a` from PTX ISA version 9.0)
* And are supported on following family-specific architectures from PTX ISA version 8.8:

  + `sm_100f` or higher in the same family
  + `sm_101f` or higher in the same family (Renamed to `sm_110f` from PTX ISA version 9.0)
* `sm_110f` or higher in the same family

Examples

```
.reg .b16 ctaMask, im2colwHalo, im2colOff;

.reg .u16 i2cOffW, i2cOffH, i2cOffD;

.reg .b64 l2CachePolicy;



cp.async.bulk.prefetch.tensor.1d.L2.global.tile  [tensorMap0, {tc0}];



@p cp.async.bulk.prefetch.tensor.2d.L2.global    [tensorMap1, {tc0, tc1}];



@p cp.async.bulk.prefetch.tensor.5d.L2.global.im2col

                      [tensorMap2, {tc0, tc1, tc2, tc3, tc4}], {i2cOffW, i2cOffH, i2cOffD};



@p cp.async.bulk.prefetch.tensor.3d.L2.global.im2col.L2::cache_hint

                      [tensorMap3, {tc0, tc1, tc2}], {i2cOffW}, policy;



cp.async.bulk.prefetch.tensor.2d.L2.global.tile::gather4 [tensorMap5, {col_idx, row_idx0, row_idx1, row_idx2, row_idx3}];



cp.async.bulk.prefetch.tensor.4d.L2.global.im2col::w::128

                      [tensorMap4, {t0, t1, t2, t3}], {im2colwHalo, im2colOff};
```