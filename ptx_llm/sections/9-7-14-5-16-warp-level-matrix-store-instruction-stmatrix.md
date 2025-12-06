##### 9.7.14.5.16. Warp-level matrix store instruction: stmatrix 

`stmatrix`

Collectively store one or more matrices to shared memory.

Syntax

```
stmatrix.sync.aligned.shape.num{.trans}{.ss}.type [p], r;



.shape  = {.m8n8, .m16n8};

.num    = {.x1, .x2, .x4};

.ss     = {.shared{::cta}};

.type   = {.b16, .b8};
```

Description

Collectively store one or more matrices across all threads in a warp to the location indicated by
the address operand `p`, in `.shared` state space. If no state space is provided, generic
addressing is used, such that the address in `p` points into `.shared` space. If the generic
address doesn’t fall in `.shared` state space, then the behavior is undefined.

The `.shape` qualifier indicates the dimensions of the matrices being loaded. Each matrix element
holds 16-bit or 8-bit data as indicated by the `.type` qualifier.

`.m16n8` shape is valid only for `.b8` type.

The values `.x1`, `.x2` and `.x4` for `.num` indicate one, two or four matrices
respectively.

The mandatory `.sync` qualifier indicates that `stmatrix` causes the executing thread to wait
until all threads in the warp execute the same `stmatrix` instruction before resuming execution.

The mandatory `.aligned` qualifier indicates that all threads in the warp must execute the same
`stmatrix` instruction. In conditionally executed code, an `stmatrix` instruction should only be
used if it is known that all threads in the warp evaluate the condition identically, otherwise the
behavior is undefined.

The behavior of `stmatrix` is undefined if all threads do not use the same qualifiers, or if any
thread in the warp has exited.

The source operand `r` is a brace-enclosed vector expression consisting of 1, 2, or 4 32-bit
registers as per the value of `.num`. Each component of the vector expression holds a fragment
from the corresponding matrix.

Supported addressing modes for `p` are described in [Addresses as Operands](#addresses-as-operands).

Consecutive instances of row need not be stored contiguously in memory. The eight addresses required
for each matrix are provided by eight threads, depending upon the value of `.num` as shown in the
following table. Each address corresponds to the start of a matrix row. Addresses addr0–addr7
correspond to the rows of the first matrix, addresses addr8–addr15 correspond to the rows of the
second matrix, and so on.

| `.num` | Threads 0–7 | Threads 8–15 | Threads 16–23 | Threads 24–31 |
| --- | --- | --- | --- | --- |
| `.x1` | addr0–addr7 | – | – | – |
| `.x2` | addr0–addr7 | addr8–addr15 | – | – |
| `.x4` | addr0–addr7 | addr8–addr15 | addr16–addr23 | addr24–addr31 |

When storing 8x8 matrices, a group of four consecutive threads stores 16 bytes. The matrix addresses
must be naturally aligned accordingly.

Each thread in a warp stores fragments of a row, with thread 0 storing the first fragment from its
register `r`, and so on. A group of four threads stores an entire row of the matrix as shown in
[Figure 107](#mma-stmatrix-fragments).

![_images/mma-stmatrix-fragments.png](_images/mma-stmatrix-fragments.png)


Figure 107 stmatrix fragment layout for one 8x8 matrix with 16-bit elements[](#mma-stmatrix-fragments "Permalink to this image")

When `.num` = `.x2`, the elements of the second matrix are storedd from the next source register
in each thread as per the layout in above table. Similarly, when `.num` = `.x4`, elements of the
third and fourth matrices are stored from the subsequent source registers in each thread.

For 16x8 matrix shape, each of the 32 threads in the warp provides four elements of data per matrix.

Each element in the source operand `r` is of type `.b32` and contains four 8 bit elements `e0`,
`e1`, `e2`, `e3` with `e0` and `e3` containing the LSB and MSB respectively of register `r`.

![_images/mma-stmatrix-fragments-168.png](_images/mma-stmatrix-fragments-168.png)


Figure 108 stmatrix fragment layout for one 16x8 matrix with 8 bit elements[](#mma-stmatrix-fragments-168 "Permalink to this image")

Optional qualifier `.trans` indicates that the matrix is stored in column-major format. However,
for 16x8 matrices, `.trans` is mandatory.

The `stmatrix` instruction is treated as a weak memory operation in the [Memory Consistency Model](#memory-consistency-model).

PTX ISA Notes

Introduced in PTX ISA version 7.8.

Support for `.m16n8` shape is introduced in PTX ISA version 8.6.

Support for `.b8` type with `stmatrix` is introduced in PTX ISA version 8.6.

Target ISA Notes

Requires `sm_90` or higher.

Shape `.m16n8` is supported on following architectures:

* `sm_100a`
* `sm_101a` (Renamed to `sm_110a` from PTX ISA version 9.0)
* `sm_120a`
* And is supported on following family-specific architectures from PTX ISA version 8.8:

  > + `sm_100f` or higher in the same family
  > + `sm_101f` or higher in the same family (Renamed to `sm_110f` from PTX ISA version 9.0)
  > + `sm_120f` or higher in the same family
* `sm_110f` or higher in the same family

Type `.b8` with `stmatrix` is supported on following architectures:

* `sm_100a`
* `sm_101a` (Renamed to `sm_110a` from PTX ISA version 9.0)
* `sm_120a`
* And is supported on following family-specific architectures from PTX ISA version 8.8:

  > + `sm_100f` or higher in the same family
  > + `sm_101f` or higher in the same family (Renamed to `sm_110f` from PTX ISA version 9.0)
  > + `sm_120f` or higher in the same family
* `sm_110f` or higher in the same family

Examples

```
// Store a single 8x8 matrix using 64-bit addressing

.reg .b64 addr;

.reg .b32 r;

stmatrix.sync.aligned.m8n8.x1.shared.b16 [addr], {r};



// Store two 8x8 matrices in column-major format

.reg .b64 addr;

.reg .b32 r<2>;

stmatrix.sync.aligned.m8n8.x2.trans.shared::cta.b16 [addr], {r0, r1};



// Store four 8x8 matrices

.reg .b64 addr;

.reg .b32 r<4>;

stmatrix.sync.aligned.m8n8.x4.b16 [addr], {r0, r1, r2, r3};



// Store a single 16x8 matrix using generic addressing

.reg .b64 addr;

.reg .b32 r;

stmatrix.sync.aligned.m16n8.x1.trans.shared.b8 [addr], {r};



// Store two 16x8 matrices

.reg .b64 addr;

.reg .b32 r<2>;

stmatrix.sync.aligned.m16n8.x2.trans.shared::cta.b8 [addr],{r0, r1};



// Store four 16x8 matrices

.reg .b64 addr;

.reg .b32 r<4>;

stmatrix.sync.aligned.m16n8.x4.b8 [addr], {r0, r1, r2, r3};
```