##### 9.7.14.5.14. Multiply-and-Accumulate Instruction: mma 

`mma`

Perform matrix multiply-and-accumulate operation

Syntax

Half precision floating point type:

```
mma.sync.aligned.m8n8k4.alayout.blayout.dtype.f16.f16.ctype  d, a, b, c;

mma.sync.aligned.m16n8k8.row.col.dtype.f16.f16.ctype  d, a, b, c;

mma.sync.aligned.m16n8k16.row.col.dtype.f16.f16.ctype d, a, b, c;



.alayout = {.row, .col};

.blayout = {.row, .col};

.ctype   = {.f16, .f32};

.dtype   = {.f16, .f32};
```

Alternate floating point type:

```
mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32        d, a, b, c;

mma.sync.aligned.m16n8k8.row.col.f32.atype.btype.f32      d, a, b, c;

mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32       d, a, b, c;

mma.sync.aligned.shape.row.col.dtype.f8type.f8type.ctype  d, a, b, c;

mma.sync.aligned.m16n8k32.row.col.kind.dtype.f8f6f4type.f8f6f4type.ctype d, a, b, c;



.atype      = {.bf16, .tf32};

.btype      = {.bf16, .tf32};

.f8type     = {.e4m3, .e5m2};

.f8f6f4type = {.e4m3, .e5m2, .e3m2, .e2m3, .e2m1};

.ctype      = {.f16, .f32};

.dtype      = {.f16, .f32};

.shape      = {.m16n8k16, .m16n8k32};

.kind       = {.kind::f8f6f4};
```

Alternate floating point type with block scaling:

```
mma.sync.aligned.m16n8k64.row.col.kind.block_scale{.scale_vec_size}.f32.e2m1.e2m1.f32.stype d, a, b, c, scale-a-data, {byte-id-a, thread-id-a}, scale-b-data, {byte-id-b, thread-id-b};



.kind           = {.kind::mxf4};

.scale_vec_size = {.scale_vec::2X};

.stype          = {.ue8m0};



mma.sync.aligned.m16n8k64.row.col.kind.block_scale.scale_vec_size.f32.e2m1.e2m1.f32.stype d, a, b, c, scale-a-data, {byte-id-a, thread-id-a}, scale-b-data, {byte-id-b, thread-id-b};



.kind           = {.kind::mxf4nvf4};

.scale_vec_size = {.scale_vec::2X, .scale_vec::4X};

.stype          = {.ue8m0, .ue4m3};



mma.sync.aligned.m16n8k32.row.col.kind.block_scale{.scale_vec_size}.f32.f8f6f4type.f8f6f4type.f32.stype d, a, b, c, scale-a-data, {byte-id-a, thread-id-a}, scale-b-data, {byte-id-b, thread-id-b};



.kind           = {.kind::mxf8f6f4};

.scale_vec_size = {.scale_vec::1X};

.f8f6f4type     = {.e4m3, .e5m2, .e3m2, .e2m3, .e2m1};

.stype          = {.ue8m0};
```

Double precision floating point type:

```
mma.sync.aligned.shape.row.col.f64.f64.f64.f64 d, a, b, c;



.shape   = {.m8n84, .m16n8k4, .m16n8k8, .m16n8k16};
```

Integer type:

```
mma.sync.aligned.shape.row.col{.satfinite}.s32.atype.btype.s32 d, a, b, c;



.shape   = {.m8n8k16, .m16n8k16, .m16n8k32}

.atype   = {.u8, .s8};

.btype   = {.u8, .s8};



mma.sync.aligned.shape.row.col{.satfinite}.s32.atype.btype.s32 d, a, b, c;



.shape   = {.m8n8k32, .m16n8k32, .m16n8k64}

.atype   = {.u4, .s4};

.btype   = {.u4, .s4};
```

Single bit:

```
mma.sync.aligned.shape.row.col.s32.b1.b1.s32.bitOp.popc d, a, b, c;



.bitOp = {.xor, .and}

.shape = {.m8n8k128, .m16n8k128, .m16n8k256}
```

Description

Perform a `MxNxK` matrix multiply and accumulate operation, `D = A*B+C`, where the A matrix is
`MxK`, the B matrix is `KxN`, and the C and D matrices are `MxN`.

Qualifier `.block_scale` specifies that the matrices A and B are scaled with `scale_A` and
`scale_B` matrices respectively before performing the matrix multiply and accumulate operation
as specified in the section [Block Scaling](#warp-level-block-scaling). The data type
corresponding to each of the element within `scale_A` and `Scale_B` matrices is specified
by `.stype`. Qualifier `.scale_vec_size` specifies the number of columns of `scale_A` matrix
and number of rows in the matrix `scale_B`.

The valid combinations of `.kind`, `.stype` and `.scale_vec_size` are described in
[Table 36](#mma-scaling-kind-type-valid-combination). For `mma` with `.kind::mxf4` when the
qualifier `.scale_vec_size` is not specified, then it defaults to `2X`. In contrast, when
`.kind` is specified as `.kind::mxf8f6f4` then the qualifier `.scale_vec_size` defaults
to `1X`. However, for `.kind::mxf4nvf4`, it is mandatory to provide valid `.scale_vec_size`.

A warp executing `mma.sync.m8n8k4` instruction computes 4 matrix multiply and accumulate
operations. Rest of the `mma.sync` operations compute a single matrix mutliply and accumulate
operation per warp.

For single-bit `mma.sync`, multiplication is replaced by a sequence of logical operations;
specifically, `mma.xor.popc` and `mma.and.popc` computes the XOR, AND respectively of a k-bit
row of A with a k-bit column of B, then counts the number of set bits in the result (`popc`). This
result is added to the corresponding element of C and written into D.

Operands `a` and `b` represent two multiplicand matrices A and B, while `c` and `d`
represent the accumulator and destination matrices, distributed across the threads in warp.
When `.block_scale` qualifier is specified, operand `scale-a-data`, `scale-b-data` represents
the scale matrix metadata corresponding to `scale_A` and `scale_B` matrices respectively. The
tuple `{byte-id-a, thread-id-a}` and `{byte-id-b, thread-id-b}` represent selectors for matrices
`scale_A` and `scale_B` respectively from their corresponding metadata arguments `scale-a-data`,
`scale-b-data`. The operands `scale-a-data`, `scale-b-data` are of type `.b32`. The operands
`byte-id-a`, `thread-id-a`, `byte-id-b`, `thread-id-b` are unsigned 16-bit integer values.
For more details on selector arguments refer [Block Scaling](#warp-level-block-scaling) section.

The registers in each thread hold a fragment of matrix as described in
[Matrix multiply-accumulate operation using mma instruction](#warp-level-matrix-instructions-for-mma).

The qualifiers `.dtype`, `.atype`, `.btype` and `.ctype` indicate the data-type of the
elements in the matrices D, A, B and C respectively. The qualifier `.stype` indicate the data-type
of the elements in the matrices `scale_A` and `scale_B`. Specific shapes have type restrictions :

* `.m8n8k4` : When `.ctype` is `.f32`, `.dtype` must also be `.f32`.
* `.m16n8k8` :

  + `.dtype` must be the same as `.ctype`.
  + `.atype` must be the same as `.btype`.

The qualifiers `.alayout` and `.blayout` indicate the row-major or column-major layouts of
matrices A and B respectively.

When `.kind` is either of `.kind::mxf8f6f4` or `.kind::f8f6f4`, the individual 4-bit and the
6-bit floating point type elements must be packed in an 8-bit container. The matrix element of type
`.e2m1` resides in central 4 bits of the 8-bit container with padding in the upper 2 bits and
lower 2 bits of the container. When the matrix element is of type `.e3m2` or `.e2m3`, the
matrix element resides in the lower 6 bits of the 8-bit container with padding in the upper 2 bits
of the container. In contrast, note that when using `mma` with `.kind::mxf4` or
`.kind::mxf4nvf4`, no explicit padding is necessary even though matrix elements are of type `.e2m1`.

Precision and rounding :
:   * `.f16` floating point operations:

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
    * `.f64` floating point operations :

      Precision of the element-wise multiplication and addition operation is identical to that of `.f64`
      precision fused multiply-add. Supported rounding modifiers are :

      + `.rn` : mantissa LSB rounds to nearest even. This is the default.
      + `.rz` : mantissa LSB rounds towards zero.
      + `.rm` : mantissa LSB rounds towards negative infinity.
      + `.rp` : mantissa LSB rounds towards positive infinity.
    * Integer operations :

      The integer `mma` operation is performed with `.s32` accumulators. The `.satfinite`
      qualifier indicates that on overflow, the accumulated value is limited to the range
      *MIN\_INT32*.. *MAX\_INT32* (where the bounds are defined as the minimum negative signed 32-bit
      integer and the maximum positive signed 32-bit integer respectively).

      If `.satfinite` is not specified, the accumulated value is wrapped instead.

The mandatory `.sync` qualifier indicates that `mma` instruction causes the executing thread to
wait until all threads in the warp execute the same `mma` instruction before resuming execution.

The mandatory `.aligned` qualifier indicates that all threads in the warp must execute the same
`mma` instruction. In conditionally executed code, a `mma` instruction should only be used if it
is known that all threads in the warp evaluate the condition identically, otherwise behavior is
undefined.

The behavior of `mma` instruction is undefined if all threads in the same warp do not use the same
qualifiers, or if any thread in the warp has exited.

Notes

Programs using double precision floating point `mma` instruction with shapes `.m16n8k4`,
`.m16n8k8`, and `.m16n8k16` require at least 64 registers for compilation.

PTX ISA Notes

Introduced in PTX ISA version 6.4.

`.f16` floating point type `mma` operation with `.m8n8k4` shape introduced in PTX ISA version
6.4.

`.f16` floating point type `mma` operation with `.m16n8k8` shape introduced in PTX ISA version
6.5.

`.u8/.s8` integer type `mma` operation with `.m8n8k16` shape introduced in PTX ISA version
6.5.

`.u4/.s4` integer type `mma` operation with `.m8n8k32` shape introduced in PTX ISA version
6.5.

`.f64` floating point type `mma` operation with `.m8n8k4` shape introduced in PTX ISA version
7.0.

`.f16` floating point type `mma` operation with `.m16n8k16` shape introduced in PTX ISA
version 7.0.

`.bf16` alternate floating point type `mma` operation with `.m16n8k8` and `.m16n8k16` shapes
introduced in PTX ISA version 7.0.

`.tf32` alternate floating point type `mma` operation with `.m16n8k4` and `.m16n8k8` shapes
introduced in PTX ISA version 7.0.

`.u8/.s8` integer type `mma` operation with `.m16n8k16` and `.m16n8k32` shapes introduced in
PTX ISA version 7.0.

`.u4/.s4` integer type `mma` operation with `.m16n8k32` and `.m16n8k64` shapes introduced in
PTX ISA version 7.0.

`.b1` single-bit integer type `mma` operation with `.m8n8k128`, `.m16n8k128` and
`.m16n8k256` shapes introduced in PTX ISA version 7.0.

Support for `.and` operation in single-bit `mma` introduced in PTX ISA version 7.1.

`.f64` floating point type `mma` operation with `.m16n8k4`, `.m16n8k8`, and `.m16n8k16`
shapes introduced in PTX ISA version 7.8.

Support for `.e4m3` and `.e5m2` alternate floating point type `mma` operation introduced in
PTX ISA version 8.4.

Support for shape `.m16n8k16` and `.f16` `dtype`/`ctype` with `.e4m3`/`.e5m2` alternate
floating point type mma operation introduced in PTX ISA version 8.7.

Support for `.e3m2`, `.e2m3`, `.e2m1` alternate floating point type `mma` operation introduced
in PTX ISA version 8.7.

Support for `.kind`, `.block_scale`, `.scale_vec_size` qualifier introduced in PTX ISA version 8.7.

Target ISA Notes

Requires `sm_70` or higher.

`.f16` floating point type `mma` operation with `.m8n8k4` shape requires `sm_70` or higher.

Note

`mma.sync.m8n8k4` is optimized for target architecture `sm_70` and may have substantially
reduced performance on other target architectures.

`.f16` floating point type `mma` operation with `.m16n8k8` shape requires `sm_75` or higher.

`.u8/.s8` integer type `mma` operation with `.m8n8k16` shape requires `sm_75` or higher.

`.u4/.s4` integer type `mma` operation with `.m8n8k32` shape `sm_75` or higher.

`.b1` single-bit integer type `mma` operation with `.m8n8k128` shape `sm_75` or higher.

`.f64` floating point type `mma` operation with `.m8n8k4` shape requires `sm_80` or higher.

`.f16` floating point type `mma` operation with `.m16n8k16` shape requires `sm_80` or
higher.

`.bf16` alternate floating point type `mma` operation with `.m16n8k8` and `.m16n8k16` shapes
requires `sm_80` or higher.

`.tf32` alternate floating point type `mma` operation with `.m16n8k4` and `.m16n8k8` shapes
requires `sm_80` or higher.

`.u8/.s8` integer type `mma` operation with `.m16n8k16` and `.m16n8k32` shapes requires
`sm_80` or higher.

`.u4/.s4` integer type `mma` operation with `.m16n8k32` and `.m16n8k64` shapes requires
`sm_80` or higher.

`.b1` single-bit integer type `mma` operation with `.m16n8k128` and `.m16n8k256` shapes
requires `sm_80` or higher.

`.and` operation in single-bit `mma` requires `sm_80` or higher.

`.f64` floating point type `mma` operation with `.m16n8k4`, `.m16n8k8`, and `.m16n8k16`
shapes require `sm_90` or higher.

`.e4m3` and `.e5m2` alternate floating point type `mma` operation requires `sm_89` or higher.

`.e3m2`, `.e2m3` and `.e2m1` alternate floating point type `mma` operation requires `sm_120a`
and is supported on `sm_120f` from PTX ISA version 8.8.

Support for `.kind`, `.block_scale`, `.scale_vec_size` qualifier requires `sm_120a` and are
supported on `sm_120f` or higher in the same family from PTX ISA version 8.8.

Examples of half precision floating point type

```
// f16 elements in C and D matrix

.reg .f16x2 %Ra<2> %Rb<2> %Rc<4> %Rd<4>

mma.sync.aligned.m8n8k4.row.col.f16.f16.f16.f16

{%Rd0, %Rd1, %Rd2, %Rd3},

{%Ra0, %Ra1},

{%Rb0, %Rb1},

{%Rc0, %Rc1, %Rc2, %Rc3};





// f16 elements in C and f32 elements in D

.reg .f16x2 %Ra<2> %Rb<2> %Rc<4>

.reg .f32 %Rd<8>

mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f16

{%Rd0, %Rd1, %Rd2, %Rd3, %Rd4, %Rd5, %Rd6, %Rd7},

{%Ra0, %Ra1},

{%Rb0, %Rb1},

{%Rc0, %Rc1, %Rc2, %Rc3};



 // f32 elements in C and D

.reg .f16x2 %Ra<2>, %Rb<1>;

.reg .f32 %Rc<4>, %Rd<4>;

mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32

  {%Rd0, %Rd1, %Rd2, %Rd3},

  {%Ra0, %Ra1},

  {%Rb0},

  {%Rc0, %Rc1, %Rc2, %Rc3};



.reg .f16x2 %Ra<4>, %Rb<2>, %Rc<2>, %Rd<2>;

mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16

  {%Rd0, %Rd1},

  {%Ra0, %Ra1, %Ra2, %Ra3},

  {%Rb0, %Rb1},

  {%Rc0, %Rc1};



.reg .f16 %Ra<4>, %Rb<2>;

.reg .f32 %Rc<2>, %Rd<2>;

mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32

  {%Rd0, %Rd1, %Rd2, %Rd3},

  {%Ra0, %Ra1, %Ra2, %Ra3},

  {%Rb0, %Rb1},

  {%Rc0, %Rc1, %Rc2, %Rc3};
```

Examples of alternate floating point type

```
.reg .b32 %Ra<2>, %Rb<1>;

.reg .f32 %Rc<4>, %Rd<4>;

mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32

  {%Rd0, %Rd1, %Rd2, %Rd3},

  {%Ra0, %Ra1},

  {%Rb0},

  {%Rc0, %Rc1, %Rc2, %Rc3};



.reg .f16x2 %Ra<2>, %Rb<1>;

.reg .f32 %Rc<4>, %Rd<4>;

mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32

  {%Rd0, %Rd1, %Rd2, %Rd3},

  {%Ra0, %Ra1},

  {%Rb0},

  {%Rc0, %Rc1, %Rc2, %Rc3};



.reg .b32 %Ra<2>, %Rb<1>;

.reg .f32 %Rc<4>, %Rd<4>;

mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32

  {%Rd0, %Rd1, %Rd2, %Rd3},

  {%Ra0, %Ra1, %Rb2, %Rb3},

  {%Rb0, %Rb1},

  {%Rc0, %Rc1, %Rc2, %Rc3};



.reg .f16x2 %Ra<2>, %Rb<1>;

.reg .f32 %Rc<4>, %Rd<4>;

mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32

  {%Rd0, %Rd1, %Rd2, %Rd3},

  {%Ra0, %Ra1, %Ra2, %Ra3},

  {%Rb0, %Rb1},

  {%Rc0, %Rc1, %Rc2, %Rc3};



.reg .b32 %Ra<4>, %Rb<4>;

.reg .f32 %Rc<4>, %Rd<4>;

mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e5m2.f32

  {%Rd0, %Rd1, %Rd2, %Rd3},

  {%Ra0, %Ra1, %Ra2, %Ra3},

  {%Rb0, %Rb1},

  {%Rc0, %Rc1, %Rc2, %Rc3};



.reg .b32 %Ra<4>, %Rb<4>;

.reg .f32 %Rc<4>, %Rd<4>;

mma.sync.aligned.m16n8k16.row.col.f32.e5m2.e4m3.f32

  {%Rd0, %Rd1, %Rd2, %Rd3},

  {%Ra0, %Ra1},

  {%Rb0},

  {%Rc0, %Rc1, %Rc2, %Rc3};



.reg .b32 %Ra<4>, %Rb<4>;

.reg .b32 %Rc<4>, %Rd<4>;

mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e5m2.f16

  {%Rd0, %Rd1},

  {%Ra0, %Ra1, %Ra2, %Ra3},

  {%Rb0, %Rb1},

  {%Rc0, %Rc1};



.reg .b32 %Ra<4>, %Rb<4>;

.reg .b32 %Rc<4>, %Rd<4>;

mma.sync.aligned.m16n8k16.row.col.f16.e5m2.e5m2.f16

  {%Rd0, %Rd1},

  {%Ra0, %Ra1},

  {%Rb0},

  {%Rc0, %Rc1};



.reg .b32 %Ra<4>, %Rb<4>;

.reg .f32 %Rc<4>, %Rd<4>;

mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e3m2.e2m3.f32

  {%Rd0, %Rd1, %Rd2, %Rd3},

  {%Ra0, %Ra1, %Ra2, %Ra3},

  {%Rb0, %Rb1},

  {%Rc0, %Rc1, %Rc2, %Rc3};



.reg .b32 %Ra<4>, %Rb<4>;

.reg .b32 %Rc<4>, %Rd<4>;

mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f16.e2m3.e2m1.f16

  {%Rd0, %Rd1},

  {%Ra0, %Ra1, %Ra2, %Ra3},

  {%Rb0, %Rb1},

  {%Rc0, %Rc1};
```

Examples of integer type

```
.reg .b32 %Ra, %Rb, %Rc<2>, %Rd<2>;



// s8 elements in A and u8 elements in B

mma.sync.aligned.m8n8k16.row.col.satfinite.s32.s8.u8.s32

  {%Rd0, %Rd1},

  {%Ra},

  {%Rb},

  {%Rc0, %Rc1};



// u4 elements in A and B matrix

mma.sync.aligned.m8n8k32.row.col.satfinite.s32.u4.u4.s32

  {%Rd0, %Rd1},

  {%Ra},

  {%Rb},

  {%Rc0, %Rc1};



// s8 elements in A and u8 elements in B

.reg .b32 %Ra<2>, %Rb, %Rc<4>, %Rd<4>;

mma.sync.aligned.m16n8k16.row.col.satfinite.s32.s8.u8.s32

  {%Rd0, %Rd1, %Rd2, %Rd3},

  {%Ra0, %Ra1},

  {%Rb},

  {%Rc0, %Rc1, %Rc2, %Rc3};



// u4 elements in A and s4 elements in B

.reg .b32 %Ra<2>, %Rb, %Rc<4>, %Rd<4>;

mma.sync.aligned.m16n8k32.row.col.satfinite.s32.u4.s4.s32

  {%Rd0, %Rd1, %Rd2, %Rd3},

  {%Ra0, %Ra1},

  {%Rb},

  {%Rc0, %Rc1, %Rc2, %Rc3};



// s8 elements in A and s8 elements in B

.reg .b32 %Ra<4>, %Rb<2>, %Rc<4>, %Rd<4>;

mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32

  {%Rd0, %Rd1, %Rd2, %Rd3},

  {%Ra0, %Ra1, %Ra2, %Ra3},

  {%Rb0, %Rb1},

  {%Rc0, %Rc1, %Rc2, %Rc3};



// u8 elements in A and u8 elements in B

.reg .b32 %Ra<4>, %Rb<2>, %Rc<4>, %Rd<4>;

mma.sync.aligned.m16n8k64.row.col.satfinite.s32.u4.u4.s32

  {%Rd0, %Rd1, %Rd2, %Rd3},

  {%Ra0, %Ra1, %Ra2, %Ra3},

  {%Rb0, %Rb1 },

  {%Rc0, %Rc1, %Rc2, %Rc3};
```

Examples of single bit type

```
// b1 elements in A and B

.reg .b32 %Ra, %Rb, %Rc<2>, %Rd<2>;

mma.sync.aligned.m8n8k128.row.col.s32.b1.b1.s32.and.popc

  {%Rd0, %Rd1},

  {%Ra},

  {%Rb},

  {%Rc0, %Rc1};



// b1 elements in A and B

.reg .b32 %Ra, %Rb, %Rc<2>, %Rd<2>;

mma.sync.aligned.m8n8k128.row.col.s32.b1.b1.s32.xor.popc

  {%Rd0, %Rd1},

  {%Ra},

  {%Rb},

  {%Rc0, %Rc1};



.reg .b32 %Ra<2>, %Rb, %Rc<4>, %Rd<4>;

mma.sync.aligned.m16n8k128.row.col.s32.b1.b1.s32.xor.popc

  {%Rd0, %Rd1, %Rd2, %Rd3},

  {%Ra0, %Ra1},

  {%Rb},

  {%Rc0, %Rc1, %Rc2, %Rc3};



.reg .b32 %Ra<2>, %Rb, %Rc<4>, %Rd<4>;

mma.sync.aligned.m16n8k128.row.col.s32.b1.b1.s32.and.popc

  {%Rd0, %Rd1, %Rd2, %Rd3},

  {%Ra0, %Ra1},

  {%Rb},

  {%Rc0, %Rc1, %Rc2, %Rc3};



.reg .b32 %Ra<4>, %Rb<2>, %Rc<4>, %Rd<4>;

mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.xor.popc

  {%Rd0, %Rd1, %Rd2, %Rd3},

  {%Ra0, %Ra1, %Ra2, %Ra3},

  {%Rb0, %Rb1},

  {%Rc0, %Rc1, %Rc2, %Rc3};



.reg .b32 %Ra<4>, %Rb<2>, %Rc<4>, %Rd<4>;

mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc

  {%Rd0, %Rd1, %Rd2, %Rd3},

  {%Ra0, %Ra1, %Ra2, %Ra3},

  {%Rb0, %Rb1},

  {%Rc0, %Rc1, %Rc2, %Rc3};
```

Examples of `.f64` floating point type

```
.reg .f64 %Ra, %Rb, %Rc<2>, %Rd<2>;

mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64

  {%Rd0, %Rd1},

  {%Ra},

  {%Rb},

  {%Rc0, %Rc1};



.reg .f64 %Ra<8>, %Rb<4>, %Rc<4>, %Rd<4>;

mma.sync.aligned.m16n8k4.row.col.f64.f64.f64.f64.rn

  {%Rd0, %Rd1, %Rd2, %Rd3},

  {%Ra0, %Ra1},

  {%Rb0},

  {%Rc0, %Rc1, %Rc2, %Rc3};



mma.sync.aligned.m16n8k8.row.col.f64.f64.f64.f64.rn

  {%Rd0, %Rd1, %Rd2, %Rd3},

  {%Ra0, %Ra1, %Ra2, %Ra3},

  {%Rb0, %Rb1},

  {%Rc0, %Rc1, %Rc2, %Rc3};



mma.sync.aligned.m16n8k16.row.col.f64.f64.f64.f64.rn

  {%Rd0, %Rd1, %Rd2, %Rd3},

  {%Ra0, %Ra1, %Ra2, %Ra3, %Ra4, %Ra5, %Ra6, %Ra7},

  {%Rb0, %Rb1, %Rb2, %Rb3},

  {%Rc0, %Rc1, %Rc2, %Rc3};
```

Examples of `mma` with block scale

```
.reg .b32 %Ra<4>, %Rb<4>;

.reg .f32 %Rc<4>, %Rd<4>;

.reg .b32 scaleAData, scaleBData;

mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.f32.e2m1.e2m1.f32.ue8m0

  {%Rd0, %Rd1, %Rd2, %Rd3},

  {%Ra0, %Ra1, %Ra2, %Ra3},

  {%Rb0, %Rb1},

  {%Rc0, %Rc1, %Rc2, %Rc3},

  scaleAData, {2, 1}, scaleBData, {2, 3};



.reg .b32 %Ra<4>, %Rb<4>;

.reg .f32 %Rc<4>, %Rd<4>;

.reg .b32 scaleAData, scaleBData;

.reg .u16 bidA, bidB, tidA, tidB;

mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3

  {%Rd0, %Rd1, %Rd2, %Rd3},

  {%Ra0, %Ra1, %Ra2, %Ra3},

  {%Rb0, %Rb1},

  {%Rc0, %Rc1, %Rc2, %Rc3},

  scaleAData, {bidA, tidA}, scaleBData, {bidB, tidB};



.reg .b32 %Ra<4>, %Rb<4>;

.reg .f32 %Rc<4>, %Rd<4>;

.reg .b32 scaleAData, scaleBData;

mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e3m2.e2m1.f32.ue8m0

  {%Rd0, %Rd1, %Rd2, %Rd3},

  {%Ra0, %Ra1, %Ra2, %Ra3},

  {%Rb0, %Rb1},

  {%Rc0, %Rc1, %Rc2, %Rc3},

  scaleAData, {0, 1}, scaleBData, {0, 1};



.reg .b32 %Ra<4>, %Rb<4>;

.reg .f32 %Rc<4>, %Rd<4>;

.reg .b32 scaleAData, scaleBData;

mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e5m2.f32.ue8m0

  {%Rd0, %Rd1, %Rd2, %Rd3},

  {%Ra0, %Ra1, %Ra2,  %Ra3},

  {%Rb0, %Rb1},

  {%Rc0, %Rc1, %Rc2, %Rc3},

  scaleAData, {0, 1}, scaleBData, {0, 0};
```