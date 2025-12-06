##### 9.7.15.6.3. Asynchronous Multiply-and-Accumulate Instruction: wgmma.mma_async.sp 

`wgmma.mma_async.sp`

Perform matrix multiply-and-accumulate operation with sparse matrix A across warpgroup

Syntax

Half precision floating point type:

```
wgmma.mma_async.sp.sync.aligned.shape.dtype.f16.f16  d, a-desc, b-desc, sp-meta, sp-sel, scale-d, imm-scale-a, imm-scale-b, imm-trans-a, imm-trans-b;



wgmma.mma_async.sp.sync.aligned.shape.dtype.f16.f16  d, a, b-desc, sp-meta, sp-sel, scale-d, imm-scale-a, imm-scale-b, imm-trans-b;



.shape   = {.m64n8k32, .m64n16k32, .m64n24k32, .m64n32k32,

            .m64n40k32, .m64n48k32, .m64n56k32, .m64n64k32,

            .m64n72k32, .m64n80k32, .m64n88k32, .m64n96k32,

            .m64n104k32, .m64n112k32, .m64n120k32, .m64n128k32,

            .m64n136k32, .m64n144k32, .m64n152k32, .m64n160k32,

            .m64n168k32, .m64n176k32, .m64n184k32, .m64n192k32,

            .m64n200k32, .m64n208k32, .m64n216k32, .m64n224k32,

            .m64n232k32, .m64n240k32, .m64n248k32, .m64n256k32};

.dtype   = {.f16, .f32};
```

Alternate floating point type :

```
.bf16 floating point type:



wgmma.mma_async.sp.sync.aligned.shape.dtype.bf16.bf16  d, a-desc, b-desc, sp-meta, sp-sel, scale-d, imm-scale-a, imm-scale-b, imm-trans-a, imm-trans-b;



wgmma.mma_async.sp.sync.aligned.shape.dtype.bf16.bf16  d, a, b-desc, sp-meta, sp-sel, scale-d, imm-scale-a, imm-scale-b, imm-trans-b;



.shape   = {.m64n8k32, .m64n16k32, .m64n24k32, .m64n32k32,

            .m64n40k32, .m64n48k32, .m64n56k32, .m64n64k32,

            .m64n72k32, .m64n80k32, .m64n88k32, .m64n96k32,

            .m64n104k32, .m64n112k32, .m64n120k32, .m64n128k32,

            .m64n136k32, .m64n144k32, .m64n152k32, .m64n160k32,

            .m64n168k32, .m64n176k32, .m64n184k32, .m64n192k32,

            .m64n200k32, .m64n208k32, .m64n216k32, .m64n224k32,

            .m64n232k32, .m64n240k32, .m64n248k32, .m64n256k32};

.dtype  = {.f32};



.tf32 floating point type:



wgmma.mma_async.sp.sync.aligned.shape.dtype.tf32.tf32  d, a-desc, b-desc, sp-meta, sp-sel, scale-d, imm-scale-a, imm-scale-b;



wgmma.mma_async.sp.sync.aligned.shape.dtype.tf32.tf32  d, a, b-desc, sp-meta, sp-sel, scale-d, imm-scale-a, imm-scale-b;



.shape   = {.m64n8k16, .m64n16k16, .m64n24k16, .m64n32k16,

            .m64n40k16, .m64n48k16, .m64n56k16, .m64n64k16,

            .m64n72k16, .m64n80k16, .m64n88k16, .m64n96k16,

            .m64n104k16, .m64n112k16, .m64n120k16, .m64n128k16,

            .m64n136k16, .m64n144k16, .m64n152k16, .m64n160k16,

            .m64n168k16, .m64n176k16, .m64n184k16, .m64n192k16,

            .m64n200k16, .m64n208k16, .m64n216k16, .m64n224k16,

            .m64n232k16, .m64n240k16, .m64n248k16, .m64n256k16};

.dtype  = {.f32};



FP8 floating point type



wgmma.mma_async.sp.sync.aligned.shape.dtype.atype.btype  d, a-desc, b-desc, sp-meta, sp-sel, scale-d, imm-scale-a, imm-scale-b;



wgmma.mma_async.sp.sync.aligned.shape.dtype.atype.btype  d, a, b-desc, sp-meta, sp-sel, scale-d, imm-scale-a, imm-scale-b;



.shape   = {.m64n8k64, .m64n16k64, .m64n24k64, .m64n32k64,

            .m64n40k64, .m64n48k64, .m64n56k64, .m64n64k64,

            .m64n72k64, .m64n80k64, .m64n88k64, .m64n96k64,

            .m64n104k64, .m64n112k64, .m64n120k64, .m64n128k64,

            .m64n136k64, .m64n144k64, .m64n152k64, .m64n160k64,

            .m64n168k64, .m64n176k64, .m64n184k64, .m64n192k64,

            .m64n200k64, .m64n208k64, .m64n216k64, .m64n224k64,

            .m64n232k64, .m64n240k64, .m64n248k64, .m64n256k64};

.atype  = {.e4m3, .e5m2};

.btype  = {.e4m3, .e5m2};

.dtype  = {.f16, .f32};
```

Integer type:

```
wgmma.mma_async.sp.sync.aligned.shape{.satfinite}.s32.atype.btype  d, a-desc, b-desc, sp-meta, sp-sel, scale-d;



wgmma.mma_async.sp.sync.aligned.shape{.satfinite}.s32.atype.btype  d, a, b-desc, sp-meta, sp-sel, scale-d;



.shape   = {.m64n8k64, .m64n16k64, .m64n24k64, .m64n32k64,

            .m64n48k64, .m64n64k64, .m64n80k64, .m64n96k64,

            .m64n112k64, .m64n128k64, .m64n144k64, .m64n160k64,

            .m64n176k64, .m64n192k64, .m64n208k64, .m64n224k64,

            .m64n240k64, .m64n256k64};

.atype  = {.s8, .u8};

.btype  = {.s8, .u8};
```

Description

Instruction `wgmma.mma_async` issues a `MxNxK` matrix multiply and accumulate operation, `D =
A*B+D`, where the A matrix is `MxK`, the B matrix is `KxN`, and the D matrix is `MxN`.

The matrix A is stored in the packed format Mx(K/2) as described in
[Sparse matrix storage](#asynchronous-warpgroup-level-sparse-matrix-storage).

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
B in shared memory respectively. The contents of a matrix descriptor must be same across all the
warps in the warpgroup. The format of the matrix descriptor is described in
[Matrix Descriptor Format](#asynchronous-warpgroup-level-matrix-shared-memory-layout-matrix-descriptor). Matrix A is
structured sparse as described in [Sparse matrix storage](#asynchronous-warpgroup-level-sparse-matrix-storage). Operands `sp-meta` and `sp-sel`
represent sparsity metadata and sparsity selector respectively. Operand `sp-meta` is a 32-bit
integer and operand `sp-sel` is a 32-bit integer constant with values in the range 0..3.

The valid values of `sp-meta` and `sp-sel` for each shape is specified in
[Sparse matrix storage](#asynchronous-warpgroup-level-sparse-matrix-storage) and are summarized here :

| Matrix shape | `.atype` | Valid values of `sp-meta` | Valid values of `sp-sel` |
| --- | --- | --- | --- |
| `.m64nNk16` | `.tf32` | 0b1110 , 0b0100 | 0 (threads T0, T1) or 1 (threads T2, T3) |
| `.m64nNk32` | `.f16`/ `.bf16` | 0b00, 0b01, 0b10, 0b11 | 0 (threads T0, T1) or 1 (threads T2, T3) |
| `.m64nNk64` | `.e4m3` / `.e5m2` / `.s8` / `.u8` | 0b00, 0b01, 0b10, 0b11 | 0 (all threads contribute) |

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

Introduced in PTX ISA version 8.2.

Support for `.u8.s8` and `.s8.u8` as .atype.btype introduced in PTX ISA version 8.4.

Target ISA Notes

Requires `sm_90a`.

Examples of integer type

```
wgmma.fence.sync.aligned;

wgmma.mma_async.sp.sync.aligned.m64n8k64.s32.u8.u8  {s32d0, s32d1, s32d2, s32d3},

                                                    descA, descB, spMeta, 0, scaleD;

wgmma.mma_async.sp.sync.aligned.m64n8k64.s32.s8.u8  {s32d0, s32d1, s32d2, s32d3},

                                                    descA, descB, spMeta, 0, scaleD;

wgmma.commit_group.sync.aligned;

wgmma.wait_group.sync.aligned 0;
```