##### 9.7.14.6.3. Multiply-and-Accumulate Instruction: mma.sp / mma.sp::ordered_metadata 

`mma.sp`, `mma.sp::ordered_metadata`

Perform matrix multiply-and-accumulate operation with sparse matrix A

Syntax

Half precision floating point type:

```
mma.spvariant.sync.aligned.m16n8k16.row.col.dtype.f16.f16.ctype  d, a, b, c, e, f;

mma.spvariant.sync.aligned.m16n8k32.row.col.dtype.f16.f16.ctype  d, a, b, c, e, f;



.ctype     = {.f16, .f32};

.dtype     = {.f16, .f32};

.spvariant = {.sp, .sp::ordered_metadata};
```

Alternate floating point type:

```
mma.spvariant.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32     d, a, b, c, e, f;

mma.spvariant.sync.aligned.m16n8k32.row.col.f32.bf16.bf16.f32     d, a, b, c, e, f;

mma.spvariant.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32      d, a, b, c, e, f;

mma.spvariant.sync.aligned.m16n8k16.row.col.f32.tf32.tf32.f32     d, a, b, c, e, f;

mma.spvariant.sync.aligned.m16n8k64.row.col.f32.f8type.f8type.f32 d, a, b, c, e, f;

mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.kind.dtype.f8f6f4type.f8f6f4type.ctype d, a, b, c, e, f;



.f8type     = {.e4m3, .e5m2};

.spvariant  = {.sp, .sp::ordered_metadata};

.f8f6f4type = {.e4m3, .e5m2, .e3m2, .e2m3, .e2m1};

.kind       = {kind::f8f6f4};

.ctype      = {.f16, .f32};

.dtype      = {.f16, .f32};
```

Alternate floating point type with block scaling:

```
mma.spvariant.sync.aligned.m16n8k128.row.col.kind.block_scale{.scale_vec_size}.f32.e2m1.e2m1.f32.stype d, a, b, c, e, f, scale-a-data, {byte-id-a, thread-id-a}, scale-b-data, {byte-id-b, thread-id-b};



.spvariant      = {.sp::ordered_metadata};

.kind           = {.kind::mxf4};

.scale_vec_size = {.scale_vec::2X};

.stype          = {.ue8m0};



mma.spvariant.sync.aligned.m16n8k128.row.col.kind.block_scale.scale_vec_size.f32.e2m1.e2m1.f32.stype d, a, b, c, e, f, scale-a-data, {byte-id-a, thread-id-a}, scale-b-data, {byte-id-b, thread-id-b};



.spvariant      = {.sp::ordered_metadata};

.kind           = {.kind::mxf4nvf4};

.scale_vec_size = {.scale_vec::2X, .scale_vec::4X};

.stype          = {.ue8m0, .ue4m3};



mma.spvariant.sync.aligned.m16n8k64.row.col.kind.block_scale{.scale_vec_size}.f32.f8f6f4type.f8f6f4type.f32.stype d, a, b, c, e, f, scale-a-data, {byte-id-a, thread-id-a}, scale-b-data, {byte-id-b, thread-id-b};



.spvariant      = {.sp::ordered_metadata};

.kind           = {.kind::mxf8f6f4};

.scale_vec_size = {.scale_vec::1X};

.f8f6f4type     = {.e4m3, .e5m2, .e3m2, .e2m3, .e2m1};

.stype          = {.ue8m0};
```

Integer type:

```
mma.spvariant.sync.aligned.shape.row.col{.satfinite}.s32.atype.btype.s32 d, a, b, c, e, f;



.shape     = {.m16n8k32, .m16n8k64}

.atype     = {.u8, .s8};

.btype     = {.u8, .s8};

.spvariant = {.sp, .sp::ordered_metadata};



mma.spvariant.sync.aligned.shape.row.col{.satfinite}.s32.atype.btype.s32 d, a, b, c, e, f;



.shape     = {.m16n8k64, .m16n8k128}

.atype     = {.u4, .s4};

.btype     = {.u4, .s4};

.spvariant = {.sp, .sp::ordered_metadata};
```

Description

Perform a `MxNxK` matrix multiply and accumulate operation, `D = A*B+C`, where the A matrix is
`MxK`, the B matrix is `KxN`, and the C and D matrices are `MxN`.

A warp executing `mma.sp.sync/mma.sp::ordered_metadata.sync` instruction compute a single matrix
multiply and accumulate operation.

Qualifier `.block_scale` specifies that the matrices `A` and `B` are scaled with `scale_A`
and `scale_B` matrices respectively before performing the matrix multiply and accumulate operation
as specified in the section [Block Scaling](#warp-level-block-scaling). The data type corresponding
to each of the element within `scale_A` and `scale_B` matrices is specified by `.stype`.
Qualifier `.scale_vec_size` specifies the number of columns of `scale_A` matrix and number of
rows in the matrix `scale_B`.

The valid combinations of `.kind`, `.stype` and `.scale_vec_size` are described in
[Table 36](#mma-scaling-kind-type-valid-combination). For `mma` with `.kind::mxf4` when the
qualifier `.scale_vec_size` is not specified, then it defaults to `2X`. In contrast,
when `.kind` is specified as `.kind::mxf8f6f4` then the qualifier `.scale_vec_size`
defaults to `1X`. However, for `.kind::mxf4nvf4`, it is mandatory to provide valid
`.scale_vec_size`.

Operands `a` and `b` represent two multiplicand matrices A and B, while `c` and `d`
represent the accumulator and destination matrices, distributed across the threads in warp. Matrix A
is structured sparse as described in [Sparse matrix storage](#warp-level-sparse-matrix-storage) Operands `e` and `f` represent sparsity
metadata and sparsity selector respectively. Operand `e` is a 32-bit integer and operand `f` is
a 32-bit integer constant with values in the range 0..3.
When `.block_scale` qualifier is specified, operand `scale-a-data`, `scale-b-data` represents
the scale matrix metadata corresponding to `scale_A` and `scale_B` matrices respectively.
The tuple `{byte-id-a, thread-id-a}` and `{byte-id-b, thread-id-b}` represent selectors for
matrices `scale_A` and `scale_B` respectively from their corresponding metadata arguments
`scale-a-data`, `scale-b-data`. The operands `scale-a-data`, `scale-b-data` are of type
`.b32`. The operands `byte-id-a`, `thread-id-a`, `byte-id-b`, `thread-id-b` are unsigned
16-bit integer values. For more details on selector arguments refer
[Block Scaling](#warp-level-block-scaling) section.

Instruction `mma.sp::ordered_metadata` requires the indices in the sparsity metadata to be sorted
in an increasing order starting from LSB, otherwise behavior is undefined.

The registers in each thread hold a fragment of matrix as described in
[Matrix fragments for multiply-accumulate operation with sparse matrix A](#warp-level-matrix-fragments-for-sparse-mma).

The qualifiers `.dtype`, `.atype`, `.btype` and `.ctype` indicate the data-type of the
elements in the matrices D, A, B and C respectively. The qualifier `.stype` indicate the
data-type of the elements in the matrices `scale_A` and `scale_B`. In case of shapes
`.m16n8k16` and `.m16n8k32`, `.dtype` must be the same as `.ctype`.

When `.kind` is either of `.kind::mxf8f6f4` or `.kind::f8f6f4`, the individual 4-bit and
the 6-bit floating point type elements must be packed in an 8-bit container. The matrix element
of type `.e2m1` resides in central 4 bits of the 8-bit container with padding in the upper 2
bits and lower 2 bits of the container. When the matrix element is of type `.e3m2` or `.e2m3`,
the matrix element resides in the lower 6 bits of the 8-bit container with padding in the upper
2 bits of the container. In contrast, note that when using `mma` with `.kind::mxf4` or
`.kind::mxf4nvf4`, no explicit padding is necessary even though matrix elements are of type
`.e2m1`.

Precision and rounding :
:   * `.f16` floating point operations :

      Element-wise multiplication of matrix A and B is performed with at least single
      precision. When `.ctype` or `.dtype` is `.f32`, accumulation of the intermediate values
      is performed with at least single precision. When both `.ctype` and `.dtype` are specified
      as `.f16`, the accumulation is performed with at least half precision.

      The accumulation order, rounding and handling of subnormal inputs are unspecified.
    * `.e4m3`, `.e5m2`, `.e3m2`, `.e2m3`, `.e2m1` floating point operations :

      Element-wise multiplication of matrix A and B is performed with specified precision. Accumulation
      of the intermediate values is performed with at least single precision.

      The accumulation order, rounding, and handling of subnormal inputs are unspecified.
    * `.bf16` and `.tf32` floating point operations :

      Element-wise multiplication of matrix A and B is performed with specified
      precision. Accumulation of the intermediate values is performed with at least single
      precision.

      The accumulation order, rounding, and handling of subnormal inputs are unspecified.
    * Integer operations :

      The integer `mma.sp/mma.sp::ordered_metadata` operation is performed with `.s32` accumulators.
      The `.satfinite` qualifier indicates that on overflow, the accumulated value is limited to the range
      *MIN\_INT32*.. *MAX\_INT32* (where the bounds are defined as the minimum negative signed 32-bit
      integer and the maximum positive signed 32-bit integer respectively).

      If `.satfinite` is not specified, the accumulated value is wrapped instead.

The mandatory `.sync` qualifier indicates that `mma.sp/mma.sp::ordered_metadata` instruction causes
the executing thread to wait until all threads in the warp execute the same `mma.sp/mma.sp::ordered_metadata`
instruction before resuming execution.

The mandatory `.aligned` qualifier indicates that all threads in the warp must execute the same
`mma.sp/mma.sp::ordered_metadata` instruction. In conditionally executed code, a `mma.sp/mma.sp::ordered_metadata`
instruction should only be used if it is known that all threads in the warp evaluate the condition identically,
otherwise behavior is undefined.

The behavior of `mma.sp/mma.sp::ordered_metadata` instruction is undefined if all threads in the same warp
do not use the same qualifiers, or if any thread in the warp has exited.

Notes

`mma.sp` instruction may have substantially reduced performance on some target architectures.
Hence, it is advised to use `mma.sp::ordered_metadata` instruction.

PTX ISA Notes

Introduced in PTX ISA version 7.1.

Support for `.e4m3` and `.e5m2` alternate floating point type `mma` operation introduced in
PTX ISA version 8.4.

`mma.sp::ordered_metadata` introduced in PTX ISA version 8.5.

Support for shape `.m16n8k32` and `.f16` dtype/ctype with `.e4m3`/`.e5m2` alternate floating
point type `mma` operation introduced in PTX ISA version 8.7.

Support for `.e3m2`, `.e2m3`, `.e2m1` alternate floating point type `mma` operation introduced
in PTX ISA version 8.7.

Support for `.kind`, `.block_scale`, `.scale_vec_size` qualifier introduced in PTX ISA version 8.7.

Target ISA Notes

Requires `sm_80` or higher.

`.e4m3` and `.e5m2` alternate floating point type `mma` operation requires `sm_89` or higher.

`mma.sp::ordered_metadata` requires `sm_80` or higher.

Support for shape `.m16n8k32` and `.f16` dtype/ctype with `.e4m3`/`.e5m2` alternate floating
point type `mma` operation requires `sm_120`.

`.e3m2`, `.e2m3` and `.e2m1` alternate floating point type `mma` operation requires
`sm_120a` and are supported on `sm_120f` or higher in the same family from PTX ISA version 8.8.

Support for `.kind`, `.block_scale`, `.scale_vec_size` qualifier requires `sm_120a` and are
supported on `sm_120f` and later generation targets in the same family from PTX ISA version 8.8 except for `.kind::mxf4nvf4`/`.kind::mxf4`.

Qualifiers `.kind::mxf4nvf4` and `.kind::mxf4` are supported on following architectures:

* `sm_120a`
* `sm_121a`

Examples of half precision floating point type

```
// f16 elements in C and D matrix

.reg .f16x2 %Ra<2> %Rb<2> %Rc<2> %Rd<2>

.reg .b32 %Re;

mma.sp.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16

  {%Rd0, %Rd1},

  {%Ra0, %Ra1},

  {%Rb0, %Rb1},

  {%Rc0, %Rc1}, %Re, 0x1;



.reg .f16x2 %Ra<2> %Rb<2> %Rc<2> %Rd<2>

.reg .b32 %Re;



mma.sp::ordered_metadata.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16

  {%Rd0, %Rd1},

  {%Ra0, %Ra1},

  {%Rb0, %Rb1},

  {%Rc0, %Rc1}, %Re, 0x1;
```

Examples of alternate floating point type

```
.reg .b32 %Ra<2>, %Rb<2>;

.reg .f32 %Rc<4>, %Rd<4>;

.reg .b32 %Re;

mma.sp.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32

  {%Rd0, %Rd1, %Rd2, %Rd3},

  {%Ra0, %Ra1},

  {%Rb0, %Rb1},

  {%Rc0, %Rc1, %Rc2, %Rc3}, %Re, 0x1;



.reg .b32 %Ra<2>, %Rb<2>;

.reg .f32 %Rc<4>, %Rd<4>;

.reg .b32 %Re;

mma.sp.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32

  {%Rd0, %Rd1, %Rd2, %Rd3},

  {%Ra0, %Ra1},

  {%Rb0, %Rb1},

  {%Rc0, %Rc1, %Rc2, %Rc3}, %Re, 0x1;



.reg .b32 %Ra<4>, %Rb<4>;

.reg .f32 %Rc<4>, %Rd<4>;

.reg .b32 %Re;

mma.sp.sync.aligned.m16n8k32.row.col.f32.bf16.bf16.f32

  {%Rd0, %Rd1, %Rd2, %Rd3},

  {%Ra0, %Ra1, %Ra2, %Ra3},

  {%Rb0, %Rb1, %Rb2, %Rb3},

  {%Rc0, %Rc1, %Rc2, %Rc3}, %Re, 0x1;



.reg .b32 %Ra<4>, %Rb<4>;

.reg .f32 %Rc<4>, %Rd<4>;

.reg .b32 %Re;

mma.sp.sync.aligned.m16n8k64.row.col.f32.e5m2.e4m3.f32

  {%Rd0, %Rd1, %Rd2, %Rd3},

  {%Ra0, %Ra1, %Ra2, %Ra3},

  {%Rb0, %Rb1, %Rb2, %Rb3},

  {%Rc0, %Rc1, %Rc2, %Rc3}, %Re, 0;



.reg .b32 %Ra<2>, %Rb<2>;

.reg .f32 %Rc<4>, %Rd<4>;

.reg .b32 %Re;

mma.sp::ordered_metadata.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32

  {%Rd0, %Rd1, %Rd2, %Rd3},

  {%Ra0, %Ra1},

  {%Rb0, %Rb1},

  {%Rc0, %Rc1, %Rc2, %Rc3}, %Re, 0x1;



.reg .b32 %Ra<4>, %Rb<4>;

.reg .f32 %Rc<4>, %Rd<4>;

.reg .b32 %Re;

mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.kind::f8f6f4.f32.e3m2.e2m3.f32

  {%Rd0, %Rd1, %Rd2, %Rd3},

  {%Ra0, %Ra1, %Ra2, %Ra3},

  {%Rb0, %Rb1, %Rb2, %Rb3},

  {%Rc0, %Rc1, %Rc2, %Rc3}, %Re, 0;



.reg .b32 %Ra<4>, %Rb<4>;

.reg .b32 %Rc<4>, %Rd<4>;

.reg .b32 %Re;

mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.kind::f8f6f4.f16.e2m3.e2m1.f16

  {%Rd0, %Rd1},

  {%Ra0, %Ra1, %Ra2, %Ra3},

  {%Rb0, %Rb1, %Rb2, %Rb3},

  {%Rc0, %Rc1}, %Re, 0;
```

Examples of integer type

```
.reg .b32 %Ra<4>, %Rb<4>, %Rc<4>, %Rd<4>;

.reg .u32 %Re;



// u8 elements in A and B matrix

mma.sp.sync.aligned.m16n8k32.row.col.satfinite.s32.u8.u8.s32

  {%Rd0, %Rd1, %Rd2, %Rd3},

  {%Ra0, %Ra1},

  {%Rb0, %Rb1},

  {%Rc0, %Rc1, %Rc2, %Rc3}, %Re, 0x1;



// s8 elements in A and B matrix

mma.sp.sync.aligned.m16n8k64.row.col.satfinite.s32.s8.s8.s32

  {%Rd0, %Rd1, %Rd2, %Rd3},

  {%Ra0, %Ra1, %Ra2, %Ra3},

  {%Rb0, %Rb1, %Rb2, %Rb3},

  {%Rc0, %Rc1, %Rc2, %Rc3}, %Re, 0x0;



// s8 elements in A and B matrix with ordered metadata

mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.satfinite.s32.s8.s8.s32

  {%Rd0, %Rd1, %Rd2, %Rd3},

  {%Ra0, %Ra1, %Ra2, %Ra3},

  {%Rb0, %Rb1, %Rb2, %Rb3},

  {%Rc0, %Rc1, %Rc2, %Rc3}, %Re, 0x0;



// u4 elements in A and B matrix

mma.sp.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32

  {%Rd0, %Rd1, %Rd2, %Rd3},

  {%Ra0, %Ra1},

  {%Rb0, %Rb1},

  {%Rc0, %Rc1, %Rc2, %Rc3}, %Re, 0x1;



// u4 elements in A and B matrix

mma.sp.sync.aligned.m16n8k128.row.col.satfinite.s32.u4.u4.s32

  {%Rd0, %Rd1, %Rd2, %Rd3},

  {%Ra0, %Ra1, %Ra2, %Ra3},

  {%Rb0, %Rb1, %Rb2, %Rb3},

  {%Rc0, %Rc1, %Rc2, %Rc3}, %Re, 0x0;
```

Examples of mma with block scale

```
.reg .b32 %Ra<4>, %Rb<4>;

.reg .f32 %Rc<4>, %Rd<4>;

.reg .b32 scaleAData, scaleBData;

.reg .b32 %Re;

mma.sp::ordered_metadata.sync.aligned.m16n8k128.row.col.kind::mxf4.block_scale.f32.e2m1.e2m1.f32.ue8m0

  {%Rd0, %Rd1, %Rd2, %Rd3},

  {%Ra0, %Ra1, %Ra2, %Ra3},

  {%Rb0, %Rb1, %Rb2, %Rb3},

  {%Rc0, %Rc1, %Rc2, %Rc3},

  %Re, 0,

  scaleAData, {2, 1}, scaleBData, {2, 3};



.reg .b32 %Ra<4>, %Rb<4>;

.reg .f32 %Rc<4>, %Rd<4>;

.reg .b32 scaleAData, scaleBData;

.reg .u16 bidA, bidB, tidA, tidB;

.reg .b32 %Re;

mma.sp::ordered_metadata.sync.aligned.m16n8k128.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3

  {%Rd0, %Rd1, %Rd2, %Rd3},

  {%Ra0, %Ra1, %Ra2, %Ra3},

  {%Rb0, %Rb1, %Rb2, %Rb3},

  {%Rc0, %Rc1, %Rc2, %Rc3},

  %Re, 0,

  scaleAData, {bidA, tidA}, scaleBData, {bidB, tidB};



.reg .b32 %Ra<4>, %Rb<4>;

.reg .f32 %Rc<4>, %Rd<4>;

.reg .b32 scaleAData, scaleBData;

.reg .b32 %Re;

mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e3m2.e2m1.f32.ue8m0

  {%Rd0, %Rd1, %Rd2, %Rd3},

  {%Ra0, %Ra1, %Ra2, %Ra3},

  {%Rb0, %Rb1, %Rb2, %Rb3},

  {%Rc0, %Rc1, %Rc2, %Rc3},

  %Re, 0,

  scaleAData, {0, 1}, scaleBData, {0, 1};



.reg .b32 %Ra<4>, %Rb<4>;

.reg .f32 %Rc<4>, %Rd<4>;

.reg .b32 scaleAData, scaleBData;

.reg .b32 %Re;

mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e5m2.f32.ue8m0

  {%Rd0, %Rd1, %Rd2, %Rd3},

  {%Ra0, %Ra1, %Ra2,  %Ra3},

  {%Rb0, %Rb1, %Rb2, %Rb3},

  {%Rc0, %Rc1, %Rc2, %Rc3},

  %Re, 0,

  scaleAData, {0, 1}, scaleBData, {0, 0};
```