##### 9.7.14.4.5. Warp-level Matrix Multiply-and-Accumulate Instruction: wmma.mma 

`wmma.mma`

Perform a single matrix multiply-and-accumulate operation across a warp

Syntax

```
// Floating point (.f16 multiplicands) wmma.mma

wmma.mma.sync.aligned.alayout.blayout.shape.dtype.ctype d, a, b, c;



// Integer (.u8/.s8 multiplicands) wmma.mma

wmma.mma.sync.aligned.alayout.blayout.shape.s32.atype.btype.s32{.satfinite} d, a, b, c;



.alayout = {.row, .col};

.blayout = {.row, .col};

.shape  =  {.m16n16k16, .m8n32k16, .m32n8k16};

.dtype   = {.f16, .f32};

.atype   = {.s8, .u8};

.btype   = {.s8, .u8};

.ctype   = {.f16, .f32};
```

Floating point format `.bf16` `wmma.mma`:

```
wmma.mma.sync.aligned.alayout.blayout.shape.f32.atype.btype.f32 d, a, b, c;

.alayout = {.row, .col};

.blayout = {.row, .col};

.shape   = {.m16n16k16, .m8n32k16, .m32n8k16};

.atype   = {.bf16 };

.btype   = {.bf16};
```

Floating point format `.tf32` `wmma.mma`:

```
wmma.mma.sync.aligned.alayout.blayout.shape.f32.atype.btype.f32 d, a, b, c;

.alayout = {.row, .col};

.blayout = {.row, .col};

.shape   = {.m16n16k8 };

.atype   = {.tf32 };

.btype   = {.tf32};
```

Floating point Double precision `wmma.mma`:

```
wmma.mma.sync.aligned.alayout.blayout.shape{.rnd}.f64.f64.f64.f64 d, a, b, c;

.alayout = {.row, .col};

.blayout = {.row, .col};

.shape   = {.m8n8k4 };

.rnd = { .rn, .rz, .rm, .rp };
```

Sub-byte (`.u4`/`.s4` multiplicands) `wmma.mma`:

```
wmma.mma.sync.aligned.row.col.shape.s32.atype.btype.s32{.satfinite} d, a, b, c;

.shape  = {.m8n8k32};

.atype  = {.s4, .u4};

.btype  = {.s4, .u4};
```

Single-bit (`.b1` multiplicands) `wmma.mma`:

```
wmma.mma.op.popc.sync.aligned.row.col.shape.s32.atype.btype.s32 d, a, b, c;

.shape  = {.m8n8k128};

.atype  = {.b1};

.btype  = {.b1};

.op     = {.xor, .and}
```

Description

Perform a warp-level matrix multiply-and-accumulate computation `D = A * B + C` using matrices A,
B and C loaded in registers `a`, `b` and `c` respectively, and store the result matrix in
register `d`. The register arguments `a`, `b`, `c` and `d` hold unspecified fragments of
the corresponding matrices as described in [Matrix Fragments for WMMA](#warp-level-matrix-fragment)

The qualifiers `.dtype`, `.atype`, `.btype` and `.ctype` indicate the data-type of the
elements in the matrices D, A, B and C respectively.

For `wmma.mma` without explicit `.atype` and `.btype`: `.atype` and `.btype` are
implicitly set to `.f16`.

For integer `wmma`, `.ctype` and `.dtype` must be specified as `.s32`. Also, the values for
`.atype` and `.btype` must be the same, i.e., either both are `.s8` or both are `.u8`.

For sub-byte single-bit `wmma`, `.ctype` and `.dtype` must be specified as `.s32`. Also, the
values for `.atype` and `.btype` must be the same; i.e., either both are `.s4`, both are
`.u4`, or both are `.b1`.

For single-bit `wmma`, multiplication is replaced by a sequence of logical operations;
specifically, `wmma.xor.popc` and `wmma.and.popc` computes the XOR, AND respectively of a
128-bit row of A with a 128-bit column of B, then counts the number of set bits in the result
(`popc`). This result is added to the corresponding element of C and written into D.

The qualifiers `.alayout` and `.blayout` must match the layout specified on the `wmma.load`
instructions that produce the contents of operands `a` and `b` respectively. Similarly, the
qualifiers `.atype`, `.btype` and `.ctype` must match the corresponding qualifiers on the
`wmma.load` instructions that produce the contents of operands `a`, `b` and `c`
respectively.

The `.shape` qualifier must match the `.shape` qualifier used on the `wmma.load` instructions
that produce the contents of all three input operands `a`, `b` and `c` respectively.

The destination operand `d` is a brace-enclosed vector expression that matches the `.shape` of
the fragment computed by the `wmma.mma` instruction.

Saturation at the output:
:   The optional qualifier `.satfinite` indicates that the final values in the destination register
    are saturated as follows:

    * The output is clamped to the minimum or maximum 32-bit signed integer value. Otherwise, if the
      accumulation would overflow, the value wraps.

Precision and rounding for `.f16` floating point operations:
:   Element-wise multiplication of matrix A and B is performed with at least single precision. When
    `.ctype` or `.dtype` is `.f32`, accumulation of the intermediate values is performed with
    at least single precision. When both `.ctype` and `.dtype` are specified as `.f16`, the
    accumulation is performed with at least half precision.

    The accumulation order, rounding and handling of subnormal inputs is unspecified.

Precision and rounding for `.bf16`, `.tf32` floating point operations:
:   Element-wise multiplication of matrix A and B is performed with specified precision. Accumulation
    of the intermediate values is performed with at least single precision.

    The accumulation order, rounding and handling of subnormal inputs is unspecified.

Rounding modifiers on double precision `wmma.mma` (default is `.rn`):

`.rn`
:   mantissa LSB rounds to nearest even

`.rz`
:   mantissa LSB rounds towards zero

`.rm`
:   mantissa LSB rounds towards negative infinity

`.rp`
:   mantissa LSB rounds towards positive infinity

The mandatory `.sync` qualifier indicates that `wmma.mma` causes the executing thread to wait
until all threads in the warp execute the same `wmma.mma` instruction before resuming execution.

The mandatory `.aligned` qualifier indicates that all threads in the warp must execute the same
`wmma.mma` instruction. In conditionally executed code, a `wmma.mma` instruction should only be
used if it is known that all threads in the warp evaluate the condition identically, otherwise
behavior is undefined.

The behavior of `wmma.mma` is undefined if all threads in the same warp do not use the same
qualifiers, or if any thread in the warp has exited.

PTX ISA Notes

Introduced in PTX ISA version 6.0.

`.m8n32k16` and `.m32n8k16` introduced in PTX ISA version 6.1.

Integer, sub-byte integer and single-bit `wmma` introduced in PTX ISA version 6.3.

Double precision and alternate floating point precision `wmma` introduced in PTX ISA version 7.0.

Support for `.and` operation in single-bit `wmma` introduced in PTX ISA version 7.1.

Modifier `.aligned` is required from PTX ISA version 6.3 onwards, and considered implicit in PTX
ISA versions less than 6.3.

Support for `.satfinite` on floating point `wmma.mma` is deprecated in PTX ISA version 6.4 and
is removed from PTX ISA version 6.5.

Preview Feature:
:   Sub-byte `wmma` and single-bit `wmma` are preview features in PTX ISA. All details are
    subject to change with no guarantees of backward compatibility on future PTX ISA versions or SM
    architectures.

Target ISA Notes

Floating point `wmma` requires `sm_70` or higher.

Integer `wmma` requires `sm_72` or higher.

Sub-byte and single-bit `wmma` requires `sm_75` or higher.

Double precision, alternate floating point precision `wmma` require `sm_80` or higher.

`.and` operation in single-bit `wmma` requires `sm_80` or higher.

Examples

```
.global .align 32 .f16 A[256], B[256];

.global .align 32 .f32 C[256], D[256];

.reg .b32 a<8> b<8> c<8> d<8>;



wmma.load.a.sync.aligned.m16n16k16.global.row.f16

        {a0, a1, a2, a3, a4, a5, a6, a7}, [A];

wmma.load.b.sync.aligned.m16n16k16.global.col.f16

        {b0, b1, b2, b3, b4, b5, b6, b7}, [B];



wmma.load.c.sync.aligned.m16n16k16.global.row.f32

        {c0, c1, c2, c3, c4, c5, c6, c7}, [C];



wmma.mma.sync.aligned.m16n16k16.row.col.f32.f32

        {d0, d1, d2, d3, d4, d5, d6, d7},

        {a0, a1, a2, a3, a4, a5, a6, a7},

        {b0, b1, b2, b3, b4, b5, b6, b7},

        {c0, c1, c2, c3, c4, c5, c6, c7};



wmma.store.d.sync.aligned.m16n16k16.global.col.f32

        [D], {d0, d1, d2, d3, d4, d5, d6, d7};



// Compute an integer WMMA:

.reg .b32  a, b<4>;

.reg .b32 c<8>, d<8>;

wmma.mma.sync.aligned.m8n32k16.row.col.s32.s8.s8.s32

        {d0, d1, d2, d3, d4, d5, d6, d7},

        {a}, {b0, b1, b2,  b3},

        {c0, c1, c2, c3, c4, c5, c6, c7};



// Compute sub-byte WMMA:

.reg .b32 a, b, c<2> d<2>

wmma.mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32

        {d0, d1}, {a}, {b}, {c0, c1};



// Compute single-bit type WMMA:

.reg .b32 a, b, c<2> d<2>

wmma.mma.xor.popc.sync.aligned.m8n8k128.row.col.s32.b1.b1.s32

        {d0, d1}, {a}, {b}, {c0, c1};



// Compute double precision wmma

.reg .f64 a, b, c<2>, d<2>;

wmma.mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64

        {d0, d1}, {a}, {b}, {c0, c1};



// Compute alternate floating point precision wmma

.reg .b32 a<2>, b<2>, c<8>, d<8>;

wmma.mma.sync.aligned.m16n16k8.row.col.f32.tf32.tf32.f32

        {d0, d1, d2, d3, d4, d5, d6, d7},

        {a0, a1, a2, a3}, {b0, b1, b2, b3},

        {c0, c1, c2, c3, c4, c5, c6, c7};
```