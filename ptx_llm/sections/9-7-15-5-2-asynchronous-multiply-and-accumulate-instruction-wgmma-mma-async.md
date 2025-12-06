##### 9.7.15.5.2. Asynchronous Multiply-and-Accumulate Instruction: wgmma.mma_async 

`wgmma.mma_async`

Perform matrix multiply-and-accumulate operation across warpgroup

Syntax

Half precision floating point type:

```
wgmma.mma_async.sync.aligned.shape.dtype.f16.f16  d, a-desc, b-desc, scale-d, imm-scale-a, imm-scale-b, imm-trans-a, imm-trans-b;



wgmma.mma_async.sync.aligned.shape.dtype.f16.f16  d, a, b-desc, scale-d, imm-scale-a, imm-scale-b, imm-trans-b;



.shape   = {.m64n8k16, .m64n16k16, .m64n24k16, .m64n32k16,

            .m64n40k16, .m64n48k16, .m64n56k16, .m64n64k16,

            .m64n72k16, .m64n80k16, .m64n88k16, .m64n96k16,

            .m64n104k16, .m64n112k16, .m64n120k16, .m64n128k16,

            .m64n136k16, .m64n144k16, .m64n152k16, .m64n160k16,

            .m64n168k16, .m64n176k16, .m64n184k16, .m64n192k16,

            .m64n200k16, .m64n208k16, .m64n216k16, .m64n224k16,

            .m64n232k16, .m64n240k16, .m64n248k16, .m64n256k16};

.dtype   = {.f16, .f32};
```

Alternate floating point type :

```
.bf16 floating point type:



wgmma.mma_async.sync.aligned.shape.dtype.bf16.bf16  d, a-desc, b-desc, scale-d, imm-scale-a, imm-scale-b, imm-trans-a, imm-trans-b;



wgmma.mma_async.sync.aligned.shape.dtype.bf16.bf16  d, a, b-desc, scale-d, imm-scale-a, imm-scale-b, imm-trans-b;



.shape   = {.m64n8k16, .m64n16k16, .m64n24k16, .m64n32k16,

            .m64n40k16, .m64n48k16, .m64n56k16, .m64n64k16,

            .m64n72k16, .m64n80k16, .m64n88k16, .m64n96k16,

            .m64n104k16, .m64n112k16, .m64n120k16, .m64n128k16,

            .m64n136k16, .m64n144k16, .m64n152k16, .m64n160k16,

            .m64n168k16, .m64n176k16, .m64n184k16, .m64n192k16,

            .m64n200k16, .m64n208k16, .m64n216k16, .m64n224k16,

            .m64n232k16, .m64n240k16, .m64n248k16, .m64n256k16};

.dtype  = {.f32};



.tf32 floating point type:



wgmma.mma_async.sync.aligned.shape.dtype.tf32.tf32  d, a-desc, b-desc, scale-d, imm-scale-a, imm-scale-b;



wgmma.mma_async.sync.aligned.shape.dtype.tf32.tf32  d, a, b-desc, scale-d, imm-scale-a, imm-scale-b;



.shape   = {.m64n8k8, .m64n16k8, .m64n24k8, .m64n32k8,

            .m64n40k8, .m64n48k8, .m64n56k8, .m64n64k8,

            .m64n72k8, .m64n80k8, .m64n88k8, .m64n96k8,

            .m64n104k8, .m64n112k8, .m64n120k8, .m64n128k8,

            .m64n136k8, .m64n144k8, .m64n152k8, .m64n160k8,

            .m64n168k8, .m64n176k8, .m64n184k8, .m64n192k8,

            .m64n200k8, .m64n208k8, .m64n216k8, .m64n224k8,

            .m64n232k8, .m64n240k8, .m64n248k8, .m64n256k8};

.dtype  = {.f32};



FP8 floating point type



wgmma.mma_async.sync.aligned.shape.dtype.atype.btype  d, a-desc, b-desc, scale-d, imm-scale-a, imm-scale-b;



wgmma.mma_async.sync.aligned.shape.dtype.atype.btype  d, a, b-desc, scale-d, imm-scale-a, imm-scale-b;



.shape   = {.m64n8k32, .m64n16k32, .m64n24k32, .m64n32k32,

            .m64n40k32, .m64n48k32, .m64n56k32, .m64n64k32,

            .m64n72k32, .m64n80k32, .m64n88k32, .m64n96k32,

            .m64n104k32, .m64n112k32, .m64n120k32, .m64n128k32,

            .m64n136k32, .m64n144k32, .m64n152k32, .m64n160k32,

            .m64n168k32, .m64n176k32, .m64n184k32, .m64n192k32,

            .m64n200k32, .m64n208k32, .m64n216k32, .m64n224k32,

            .m64n232k32, .m64n240k32, .m64n248k32, .m64n256k32};

.atype  = {.e4m3, .e5m2};

.btype  = {.e4m3, .e5m2};

.dtype  = {.f16, .f32};
```

Integer type:

```
wgmma.mma_async.sync.aligned.shape{.satfinite}.s32.atype.btype  d, a-desc, b-desc, scale-d;



wgmma.mma_async.sync.aligned.shape{.satfinite}.s32.atype.btype  d, a, b-desc, scale-d;



.shape   = {.m64n8k32, .m64n16k32, .m64n24k32, .m64n32k32,

            .m64n48k32, .m64n64k32, .m64n80k32, .m64n96k32,

            .m64n112k32, .m64n128k32, .m64n144k32, .m64n160k32,

            .m64n176k32, .m64n192k32, .m64n208k32, .m64n224k32};

.atype  = {.s8, .u8};

.btype  = {.s8, .u8};
```

Single bit:

```
wgmma.mma_async.sync.aligned.shape.s32.b1.b1.op.popc  d, a-desc, b-desc, scale-d;



wgmma.mma_async.sync.aligned.shape.s32.b1.b1.op.popc  d, a, b-desc, scale-d;



.shape   = {.m64n8k256, .m64n16k256, .m64n24k256, .m64n32k256,

            .m64n48k256, .m64n64k256, .m64n80k256, .m64n96k256,

            .m64n112k256, .m64n128k256, .m64n144k256, .m64n160k256,

            .m64n176k256, .m64n192k256, .m64n208k256, .m64n224k256,

            .m64n240k256, .m64n256k256};

.op  = {.and};
```

Description

Instruction `wgmma.mma_async` issues a `MxNxK` matrix multiply and accumulate operation, `D =
A*B+D`, where the A matrix is `MxK`, the B matrix is `KxN`, and the D matrix is `MxN`.

The operation of the form `D = A*B` is issued when the input predicate argument `scale-d` is
false.

`wgmma.fence` instruction must be used to fence the register accesses of `wgmma.mma_async`
instruction from their prior accesses. Otherwise, the behavior is undefined.

`wgmma.commit_group` and `wgmma.wait_group` operations must be used to wait for the completion
of the asynchronous matrix multiply and accumulate operations before the results are accessed.

Register operand `d` represents the accumulator matrix as well as the destination matrix,
distributed across the participating threads. Register operand `a` represents the multiplicand
matrix A in register distributed across the participating threads. The 64-bit register operands
`a-desc` and `b-desc` are the matrix descriptors which represent the multiplicand matrices A and
B in shared memory respectively. The contents of a matrix descriptor must be same across all the warps
in the warpgroup. The format of the matrix descriptor is described in
[Matrix Descriptor Format](#asynchronous-warpgroup-level-matrix-shared-memory-layout-matrix-descriptor).

Matrices A and B are stored in row-major and column-major format respectively. For certain floating
point variants, the input matrices A and B can be transposed by specifying the value 1 for the
immediate integer arguments `imm-trans-a` and `imm-trans-b` respectively. A value of 0 can be
used to avoid the transpose operation. The valid values of `imm-trans-a` and `imm-trans-b` are 0
and 1. The transpose operation is only supported for the `wgmma.mma_async` variants with `.f16`/
`.bf16` types on matrices accessed from shared memory using matrix descriptors.

For the floating point variants of the `wgmma.mma_async` operation, each element of the input
matrices A and B can be negated by specifying the value -1 for operands `imm-scale-a` and
`imm-scale-b` respectively. A value of 1 can be used to avoid the negate operation. The valid
values of `imm-scale-a` and `imm-scale-b` are -1 and 1.

The qualifiers `.dtype`, `.atype` and `.btype` indicate the data type of the elements in
matrices D, A and B respectively. `.atype` and `.btype` must be the same for all floating point
`wgmma.mma_async` variants except for the FP8 floating point variants. The sizes of individual
data elements of matrices A and B in alternate floating point variants of the `wgmma.mma_async`
operation are as follows:

* Matrices A and B have 8-bit data elements when `.atype`/ `.btype` is `.e4m3`/`.e5m2`.
* Matrices A and B have 16-bit data elements when `.atype`/ `.btype` is `.bf16`.
* Matrices A and B have 32-bit data elements when `.atype`/ `.btype` is `.tf32`.

Precision and rounding:

* Floating point operations:

  Element-wise multiplication of matrix A and B is performed with at least single precision. When
  `.dtype` is `.f32`, accumulation of the intermediate values is performed with at least single
  precision. When `.dtype` is `.f16`, the accumulation is performed with at least half
  precision.

  The accumulation order, rounding and handling of subnormal inputs are unspecified.
* `.bf16` and `.tf32` floating point operations:

  Element-wise multiplication of matrix A and B is performed with specified
  precision. `wgmma.mma_async` operation involving type `.tf32` will truncate lower 13 bits of
  the 32-bit input data before multiplication is issued. Accumulation of the intermediate values is
  performed with at least single precision.

  The accumulation order, rounding, and handling of subnormal inputs are unspecified.
* Integer operations:

  The integer `wgmma.mma_async` operation is performed with `.s32` accumulators. The
  `.satfinite` qualifier indicates that on overflow, the accumulated value is limited to the
  range *MIN\_INT32*.. *MAX\_INT32* (where the bounds are defined as the minimum negative signed
  32-bit integer and the maximum positive signed 32-bit integer respectively).

  If `.satfinite` is not specified, the accumulated value is wrapped instead.

The mandatory `.sync` qualifier indicates that `wgmma.mma_async` instruction causes the
executing thread to wait until all threads in the warp execute the same `wgmma.mma_async`
instruction before resuming execution.

The mandatory `.aligned` qualifier indicates that all threads in the warpgroup must execute the
same `wgmma.mma_async` instruction. In conditionally executed code, a `wgmma.mma_async`
instruction should only be used if it is known that all threads in the warpgroup evaluate the
condition identically, otherwise behavior is undefined.

PTX ISA Notes

Introduced in PTX ISA version 8.0.

Support for `.u8.s8` and `.s8.u8` as .atype.btype introduced in PTX ISA version 8.4.

Target ISA Notes

Requires `sm_90a`.

Examples of half precision floating point type

```
.reg .f16x2 f16a<40>, f16d<40>;

.reg .f32   f32d<40>;

.reg .b64   descA, descB;

.reg .pred  scaleD;

wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16

  {f32d0, f32d1, f32d2, f32d3},

  {f16a0, f16a1, f16a2, f16a3},

  descB,

  1, -1, -1, 1;



wgmma.mma_async.sync.aligned.m64n72k16.f16.f16.f16

  {f16d0, f16d1,  f16d2,  f16d3,  f16d4,  f16d5,  f16d6,  f16d7,  f16d8,

   f16d9, f16d10, f16d11, f16d12, f16d13, f16d14, f16d15, f16d16, f16d17},

  descA,

  descB,

  scaleD, -1, 1, 1, 0;
```

Examples of alternate floating point type

```
.reg .f32   f32d<40>;

.reg .b32   bf16a<40>

.reg .b64   descA, descB;



wgmma.mma_async.sync.aligned.m64n120k16.f32.bf16.bf16

  {f32d0, f32d1, f32d2, f32d3, f32d4, f32d5, f32d6, f32d7, f32d8, f32d9,

   f32d10, f32d11, f32d12, f32d13, f32d14, f32d15, f32d16, f32d17, f32d18, f32d19,

   f32d20, f32d21, f32d22, f32d23, f32d24, f32d25, f32d26, f32d27, f32d28, f32d29,

   f32d30, f32d31, f32d32, f32d33, f32d34, f32d35, f32d36, f32d37, f32d38, f32d39,

   f32d40, f32d41, f32d42, f32d43, f32d44, f32d45, f32d46, f32d47, f32d48, f32d49,

   f32d50, f32d51, f32d52, f32d53, f32d54, f32d55, f32d56, f32d57, f32d58, f32d59},

  {bf16a0, bf16a1, bf16a2, bf16a3},

  descB,

  scaleD, -1, -1, 0;



.reg .f32   f32d<40>;

.reg .b64   descA, descB;



wgmma.mma_async.sync.aligned.m64n16k8.f32.tf32.tf32

  {f32d0, f32d1, f32d2, f32d3, f32d4, f32d5, f32d6, f32d7},

  descA,

  descB,

  0, -1, -1;



.reg .b32 f16d<8>, f16a<8>;

.reg .f32 f32d<8>;

.reg .b64   descA, descB;



wgmma.mma_async.sync.aligned.m64n8k32.f16.e4m3.e5m2

  {f16d0, f16d1},

  descA,

  descB,

  scaleD, -1, 1;



wgmma.mma_async.sync.aligned.m64n8k32.f32.e5m2.e4m3

  {f32d0, f32d1, f32d2, f32d3},

  {f16a0, f16a1, f16a2, f16a3},

  descB,

  1, -1, -1;
```

Examples of integer type

```
.reg .s32 s32d<8>, s32a<8>;

.reg .u32 u32a<8>;

.reg .pred scaleD;

.reg .b64   descA, descB;



wgmma.mma_async.sync.aligned.m64n8k32.s32.s8.s8.satfinite

  {s32d0, s32d1, s32d2, s32d3},

  {s32a0, s32a1, s32a2, s32a3},

  descB,

  1;



wgmma.mma_async.sync.aligned.m64n8k32.s32.u8.u8

  {s32d0, s32d1, s32d2, s32d3},

  descA,

  descB,

  scaleD;



wgmma.mma_async.sync.aligned.m64n8k32.s32.s8.u8.satfinite

  {s32d0, s32d1, s32d2, s32d3},

  {s32a0, s32a1, s32a2, s32a3},

  descB,

  scaleD;



wgmma.mma_async.sync.aligned.m64n8k32.s32.u8.s8

  {s32d0, s32d1, s32d2, s32d3},

  descA,

  descB,

  scaleD;
```

Examples of single bit type

```
.reg .s32 s32d<4>;

.reg .b32 b32a<4>;

.reg .pred scaleD;

.reg .b64   descA, descB;





wgmma.mma_async.sync.aligned.m64n8k256.s32.b1.b1.and.popc

  {s32d0, s32d1, s32d2, s32d3},

  {b32a0, b32a1, b32a2, b32a3},

  descB,

  scaleD;
```