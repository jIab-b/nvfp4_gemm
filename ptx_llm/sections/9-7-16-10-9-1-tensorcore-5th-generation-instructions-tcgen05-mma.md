###### 9.7.16.10.9.1. TensorCore 5th Generation Instructions: tcgen05.mma 

`tcgen05.mma`

Perform the 5th generation of matrix multiply and accumulate operation.

Syntax

```
// 1. Floating-point type without block scaling:



tcgen05.mma.cta_group.kind   [d-tmem],  a-desc,  b-desc, idesc,

                             { disable-output-lane }, enable-input-d {, scale-input-d};



tcgen05.mma.cta_group.kind   [d-tmem], [a-tmem], b-desc, idesc,

                             { disable-output-lane }, enable-input-d {, scale-input-d};



.kind      = { .kind::f16, .kind::tf32, .kind::f8f6f4 }

.cta_group = { .cta_group::1, .cta_group::2 }



----------------------------------------------------------------------------------



// 2. Floating-point type with block scaling:



tcgen05.mma.cta_group.kind.block_scale{.scale_vectorsize}

                                        [d-tmem],  a-desc,  b-desc, idesc,

                                        [scale-A-tmem], [scale-B-tmem], enable-input-d;



tcgen05.mma.cta_group.kind.block_scale{.scale_vectorsize}

                                        [d-tmem], [a-tmem], b-desc, idesc,

                                        [scale-A-tmem], [scale-B-tmem], enable-input-d;



.kind = { .kind::mxf8f6f4, .kind::mxf4, .kind::mxf4nvf4 }

.cta_group      = { .cta_group::1,   .cta_group::2 }

.scale_vectorsize = { .scale_vec::1X, .scale_vec::2X, .scale_vec::4X, .block16, .block32 }



----------------------------------------------------------------------------------



// 3. Convolution MMA for floating-point type without block scaling:



tcgen05.mma.cta_group.kind.collector_usage [d-tmem],  a-desc,  b-desc, idesc,

                                           { disable-output-lane }, enable-input-d {, scale-input-d};



tcgen05.mma.cta_group.kind{.ashift}.collector_usage [d-tmem], [a-tmem], b-desc, idesc,

                                                    { disable-output-lane }, enable-input-d {, scale-input-d};



tcgen05.mma.cta_group.kind.ashift{.collector_usage} [d-tmem], [a-tmem], b-desc, idesc,

                                                    { disable-output-lane }, enable-input-d {, scale-input-d};



.kind      = { .kind::f16, .kind::tf32, .kind::f8f6f4 }

.cta_group = { .cta_group::1,   .cta_group::2 }

.collector_usage = { .collector::buffer::op }

::buffer         = { ::a }

::op             = { ::fill, ::use, ::lastuse, ::discard* }



----------------------------------------------------------------------------------



// 4. Activation Stationary MMA for floating-point type with block scaling:



tcgen05.mma.cta_group.kind.block_scale{.scale_vectorsize}.collector_usage

                                            [d-tmem],  a-desc,  b-desc, idesc,

                                            [scale-A-tmem], [scale-B-tmem], enable-input-d;



tcgen05.mma.cta_group.kind.block_scale{.scale_vectorsize}.collector_usage

                                            [d-tmem], [a-tmem], b-desc, idesc,

                                            [scale-A-tmem], [scale-B-tmem], enable-input-d;



.cta_group       = { .cta_group::1,   .cta_group::2 }

.scale_vectorsize  = { .scale_vec::1X, .scale_vec::2X, .scale_vec::4X, .block16, .block32 }

.kind            = { .kind::mxf8f6f4, .kind::mxf4, .kind::mxf4nvf4 }

.collector_usage = { .collector::buffer::op }

::buffer         = { ::a }

::op             = { ::fill, ::use, ::lastuse, ::discard* }



----------------------------------------------------------------------------------



// 5. Integer type:



tcgen05.mma.cta_group.kind::i8  [d-tmem],  a-desc,  b-desc, idesc,

                                { disable-output-lane }, enable-input-d;



tcgen05.mma.cta_group.kind::i8  [d-tmem], [a-tmem], b-desc, idesc,

                                { disable-output-lane }, enable-input-d;



.cta_group = { .cta_group::1,   .cta_group::2  }



----------------------------------------------------------------------------------



// 6. Convolution MMA for integer type:



tcgen05.mma.cta_group.kind::i8.collector_usage          [d-tmem],  a-desc,  b-desc, idesc,

                                                        { disable-output-lane }, enable-input-d;



tcgen05.mma.cta_group.kind::i8.ashift{.collector_usage} [d-tmem], [a-tmem], b-desc, idesc,

                                                        { disable-output-lane }, enable-input-d;



tcgen05.mma.cta_group.kind::i8{.ashift}.collector_usage [d-tmem], [a-tmem], b-desc, idesc,

                                                        { disable-output-lane }, enable-input-d;



.cta_group       = { .cta_group::1,   .cta_group::2  }

.collector_usage = { .collector::buffer::op }

::buffer         = { ::a }

::op             = { ::fill, ::use, ::lastuse, ::discard* }
```

Description

Instruction `tcgen05.mma` is an asynchronous instruction which initiates an *MxNxK* matrix
multiply and accumulate operation,
`D = A*B+D`
where the `A` matrix is *MxK*, the `B` matrix is *KxN*, and the `D` matrix is *MxN*.

The operation of the form
`D = A*B`
is issued when the input predicate argument `enable-input-d` is false.

The optional immediate argument `scale-input-d` can be specified to scale the input
matrix `D` as follows:
`D = A*B+D * (2 ^ - scale-input-d)`

The valid range of values for argument `scale-input-d` is [0, 15]. The argument
`scale-input-d` is only valid for `.kind::tf32` and `.kind::f16`.

The 32-bit register operand `idesc` is the instruction descriptor as described
in [Instruction descriptor](#tcgen05-instruction-descriptor), specifies
the shapes, exact types, sparsity and other details of the input matrices,
output matrix and the matrix multiply and accumulate operation.

The qualifier `.cta_group::1` specifies that the matrix multiply and
accumulate operation is performed on the [Tensor Memory](#tensor-memory) of the
executing thread’s CTA only. The qualifier `.cta_group::2` specifies that the matrix
multiply and accumulate operation is performed on the [Tensor Memory](#tensor-memory)
of the executing thread’s CTA and its [peer CTA](#tcgen05-peer-cta).

All `tcgen05` instructions within a kernel must specify the same value for the `.cta_group`
qualifier.

The instruction `tcgen05.mma` has single thread semantics, unlike the collective
instructions `mma.sync` or `wgmma.mma_async`. So, a single thread issuing the
`tcgen05.mma` will result in the initiation of the whole matrix multiply and
accumulate operation. Refer to the section [Issue Granularity](#tcgen05-issue-granularity).

The qualifier `.kind` specifies the general kind of the element types of the multiplicand
matrices. The exact types of the elements of the input and output matrices for each MMA-kind
are specified in the [Instruction descriptor](#tcgen05-instruction-descriptor).

The address operand `d-tmem` specifies the address of the destination and the accumulation
matrix `D` in the [Tensor Memory](#tensor-memory). The address operand `a-tmem`
specifies the address of the matrix `A` in the [Tensor Memory](#tensor-memory).
The 64-bit register operand `a-desc` and `b-desc` are the matrix descriptors which
represent the matrices `A` and `B` in shared memory respectively. The format of the
matrix descriptor is described in [Matrix Descriptors](#tcgen05-matrix-descriptors).

The vector operand `disable-output-lane` specifies the lane(s) in the
[Tensor Memory](#tensor-memory) that should be not be updated with the resultant
matrix `D`. Elements of the vector operand `disable-output-lane` forms a mask where
each bit corresponds to a lane of the [Tensor Memory](#tensor-memory), with least
significant bit of the first element of the vector (leftmost in syntax) corresponding
to the lane 0 of the [Tensor Memory](#tensor-memory). If a bit in the mask is 1,
then the corresponding lane in the Tensor Memory for the resultant matrix `D` will not
be updated. The size of the vector is as follows:

| .cta\_group | Size of the vector disable-output-lane |
| --- | --- |
| ::1 | 4 |
| ::2 | 8 |

Qualifier `.block_scale` specifies that the matrices `A` and `B` are scaled with
`scale_A` and `scale_B` matrices respectively before performing the matrix multiply
and accumulate operation as specified in the section [Block Scaling](#tcgen05-block-scaling).
The address operand `scale-A-tmem` and `scale-B-tmem` specify the base address the
matrices `scale_A` and `scale_B` respectively in the [Tensor Memory](#tensor-memory).

For qualifier `.scale_vectorsize`,

* If `.scale_vec::NX` is specified: N specifies the number of columns in `scale_A`
  matrix and number of rows in `scale_B` matrix.
* If `.blockN` is specified: N specifies the block size for which single scale factor
  will be applied. In this form, value of N is same as the K-dimension / (N of `.scale_vec::NX`).

Aliased `.scale_vectorsize` variants:

1. `.block16` is aliased with:

   1. `.scale_vec::4X` when `.kind = .kind::mxf4nvf4` and K = 64 or 128
2. `.block32` is aliased with:

   1. `.scale_vec::1X` when `.kind = .kind::mxf8f6f4` for all supported values of K
   2. `.scale_vec::2X` when `.kind = .kind::mxf4` or `.kind::mxf4nvf4` and K = 64 or 128

The valid combinations of MMA-kind and `.scale_vectorsize` are
described in [Table 54](#tcgen05-mma-scale-valid-comb). For `.kind::mxf4` when the qualifier
`.scale_vectorsize` is not specified, then it defaults to `.block32`. For `.kind::mxf4nvf4`,
the qualifier `.scale_vectorsize` must be explicitly specified.

The qualifier `.ashift` shifts the rows of the `A` matrix down by one row, except for
the last row in the [Tensor Memory](#tensor-memory). Qualifier `.ashift` is only allowed
with *M* = 128 or *M* = 256.

The qualifier `.collector_usage` specifies the usage of collector buffer for matrix `A`.
Following collector buffer operations can be specified:

| .collector\_usage | Semantics |
| --- | --- |
| `.collector::a::fill` | Specifies that the `A` matrix read from the memory should be filled in collector buffer. |
| `.collector::a::use` | Specifies that the `A` matrix can be read from the collector buffer. This requires a previous fill to the collector buffer to be still valid. |
| `.collector::a::lastuse` | Specifies that the `A` matrix can be read from the collector buffer and the contents of the collector buffer can be discarded. This requires a previous fill to the collector buffer to be valid till the collector buffer is read. |
| `.collector::a::discard` | Specifies that the contents of the collector buffer for `A` can be discarded. |

If no `.collector_usage` qualifier is specified, then it defaults to `.collector::a::discard`.
It is illegal to specify either of `.collector::a::use` or `.collector::a::fill` along with
`.ashift`.

PTX ISA Notes

Introduced in PTX ISA version 8.6.

Qualifier `.kind::mxf4nvf4` introduced in PTX ISA version 8.7.

Qualifiers `.block16` and `.block32` introduced in PTX ISA version 8.8.

Target ISA Notes

Supported on following architectures:

* `sm_100a`
* `sm_101a` (Renamed to `sm_110a` from PTX ISA version 9.0)
* And is supported on following family-specific architectures from PTX ISA version 8.8 except `.kind::i8`:

  + `sm_100f` or higher in the same family
  + `sm_101f` or higher in the same family (Renamed to `sm_110f` from PTX ISA version 9.0)
* `sm_110f` or higher in the same family

Qualifier `.kind::i8` is supported on following architectures:

* `sm_100a`
* `sm_101a` (Renamed to `sm_110a` from PTX ISA version 9.0)
* `sm_110a`

Argument `scale-input-d` requires `sm_100a` and is supported on `sm_100f` or higher in the same family from PTX ISA version 8.8.

For `.scale_vectorsize`,

* `.scale_vec::1X`, `.scale_vec::2X`, `.scale_vec::4X` requires `sm_100a`.
* `.block16`, `.block32` requires `sm_100f` or `sm_110f`.

For Target ISA details on matrix shape, check [Target ISA Note](#tcgen05-matrix-shape-target-isa-note).

For Target ISA details on shared memory descriptor, check [Target ISA Note](#tcgen05-shared-memory-descriptor-target-isa-note).

Examples

```
tcgen05.mma.cta_group::1.kind::tf32      [taddr0],  adesc,  bdesc, idesc, {m0, m1, m2, m3}, p;

tcgen05.mma.cta_group::1.kind::mxf8f6f4  [taddr2],  [taddr1],  bdesc, idesc,

                                         [tmem_scaleA], [tmem_scaleB], p;



tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [mbarObj0];



loop:

mbarrier.try_wait.parity.b64 p, [mbarObj0], 0;

@!p bra loop;
```