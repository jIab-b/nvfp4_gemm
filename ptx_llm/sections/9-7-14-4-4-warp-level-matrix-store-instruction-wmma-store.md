##### 9.7.14.4.4. Warp-level Matrix Store Instruction: wmma.store 

`wmma.store`

Collectively store a matrix into memory for WMMA

Syntax

```
wmma.store.d.sync.aligned.layout.shape{.ss}.type [p], r {, stride};



.layout = {.row, .col};

.shape  = {.m16n16k16, .m8n32k16, .m32n8k16};

.ss     = {.global, .shared{::cta}};

.type   = {.f16, .f32, .s32};



wmma.store.d.sync.aligned.layout.shape{.ss}.type [p], r {, stride}

.layout = {.row, .col};

.shape  = {.m8n8k32, .m8n8k128};

.ss     = {.global, .shared{::cta}};

.type   = {.s32};



wmma.store.d.sync.aligned.layout.shape{.ss}.type [p], r {, stride}

.layout = {.row, .col};

.shape  = {.m16n16k8};

.ss     = {.global, .shared{::cta}};

.type   = {.f32};



wmma.store.d.sync.aligned.layout.shape{.ss}.type [p], r {, stride}

.layout = {.row, .col};

.shape  = {.m8n8k4 };

.ss     = {.global, .shared{::cta}};

.type   = {.f64};
```

Description

Collectively store a matrix across all threads in a warp at the location indicated by address
operand `p` in the specified state space from source register `r`.

If no state space is given, perform the memory accesses using
[Generic Addressing](#generic-addressing). `wmma.load` operation may be used only with `.global` and
`.shared` spaces and with generic addressing, where the address points to `.global` or
`.shared` space.

The source operand `r` is a brace-enclosed vector expression that matches the shape of the
fragment expected by the store operation, as described in [Matrix Fragments for WMMA](#warp-level-matrix-fragment).

The `.shape` qualifier indicates the dimensions of all the matrix arguments involved in the
intended `wmma` computation. It must match the `.shape` qualifier specified on the `wmma.mma`
instruction that produced the D matrix being stored.

The `.layout` qualifier indicates whether the matrix to be loaded is stored in *row-major* or
*column-major* format.

`stride` is an optional 32-bit integer operand that provides an offset in terms of matrix elements
between the start of consecutive instances of the *leading dimension* (rows or columns). The default
value of `stride` is described in
[Matrix Storage for WMMA](#warp-level-matrix-storage) and must be specified if the actual value is larger than
the default. For example, if the matrix is a sub-matrix of a larger matrix, then the value of stride
is the leading dimension of the larger matrix. Specifying a value lower than the default value
results in undefined behavior.

The required alignment for address `p` and `stride` is described in the
[Matrix Storage for WMMA](#warp-level-matrix-storage).

The mandatory `.sync` qualifier indicates that `wmma.store` causes the executing thread to wait
until all threads in the warp execute the same `wmma.store` instruction before resuming execution.

The mandatory `.aligned` qualifier indicates that all threads in the warp must execute the same
`wmma.store` instruction. In conditionally executed code, a `wmma.store` instruction should only
be used if it is known that all threads in the warp evaluate the condition identically, otherwise
behavior is undefined.

The behavior of `wmma.store` is undefined if all threads do not use the same qualifiers and the
same values of `p` and `stride`, or if any thread in the warp has exited.

`wmma.store` is treated as a *weak* memory operation in the [Memory Consistency Model](#memory-consistency-model).

PTX ISA Notes

Introduced in PTX ISA version 6.0.

`.m8n32k16` and `.m32n8k16` introduced in PTX ISA version 6.1.

Integer, sub-byte integer and single-bit `wmma` introduced in PTX ISA version 6.3.

`.m16n16k8` introduced in PTX ISA version 7.0.

Double precision `wmma` introduced in PTX ISA version 7.0.

Modifier `.aligned` is required from PTX ISA version 6.3 onwards, and considered implicit in PTX
ISA versions less than 6.3.

Support for `::cta` sub-qualifier introduced in PTX ISA version 7.8.

Preview Feature:
:   Sub-byte `wmma` and single-bit `wmma` are preview features in PTX ISA version 6.3. All
    details are subject to change with no guarantees of backward compatibility on future PTX ISA
    versions or SM architectures.

Target ISA Notes

Floating point `wmma` requires `sm_70` or higher.

Integer `wmma` requires `sm_72` or higher.

Sub-byte and single-bit `wmma` requires `sm_75` or higher.

Double precision `wmma` and shape `.m16n16k8` requires `sm_80` or higher.

Examples

```
// Storing f32 elements computed by a wmma.mma

.reg .b32 x<8>;



wmma.mma.sync.m16n16k16.row.col.f32.f32

              {d0, d1, d2, d3, d4, d5, d6, d7}, ...;

wmma.store.d.sync.m16n16k16.row.f32

              [ptr], {d0, d1, d2, d3, d4, d5, d6, d7};



// Store s32 accumulator for m16n16k16 shape:

.reg .b32 d<8>;

wmma.store.d.sync.aligned.m16n16k16.row.s32

              [ptr], {d0, d1, d2, d3, d4, d5, d6, d7};



// Store s32 accumulator for m8n8k128 shape:

.reg .b32 d<2>

wmma.store.d.sync.aligned.m8n8k128.row.s32

[ptr], {d0, d1};



// Store f64 accumulator for m8n8k4 shape:

.reg .f64 d<2>;

wmma.store.d.sync.aligned.m8n8k4.row.f64

              [ptr], {d0, d1};
```