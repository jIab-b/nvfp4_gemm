##### 9.7.14.4.1. Matrix Fragments for WMMA 

Each thread in the warp holds a fragment of the matrix. The distribution of fragments loaded by the
threads in a warp is unspecified and is target architecture dependent, and hence the identity of the
fragment within the matrix is also unspecified and is target architecture dependent. The fragment
returned by a `wmma` operation can be used as an operand for another `wmma` operation if the
shape, layout and element type of the underlying matrix matches. Since fragment layout is
architecture dependent, using the fragment returned by a `wmma` operation in one function as an
operand for a `wmma` operation in a different function may not work as expected if the two
functions are linked together but were compiled for different link-compatible SM architectures. Note
passing `wmma` fragment to a function having `.weak` linkage is unsafe since at link time
references to such function may get resolved to a function in different compilation module.

Each fragment is a vector expression whose contents are determined as follows. The identity of
individual matrix elements in the fragment is unspecified.

Integer fragments

Multiplicands (A or B):

| Data-type | Shape | Matrix | Fragment |
| --- | --- | --- | --- |
| `.u8` or `.s8` | `.m16n16k16` | A | A vector expression of two `.b32` registers, with each register containing four elements from the matrix. |
| B | A vector expression of two `.b32` registers, with each register containing four elements from the matrix. |
|  | `.m8n32k16` | A | A vector expression containing a single `.b32` register containing four elements from the matrix. |
| B | A vector expression of four `.b32` registers, with each register containing four elements from the matrix. |
|  | `.m32n8k16` | A | A vector expression of four `.b32` registers, with each register containing four elements from the matrix. |
| B | A vector expression containing single `.b32` register, with each containing four elements from the matrix. |

Accumulators (C or D):

| Data-type | Shape | Fragment |
| --- | --- | --- |
| `.s32` | `.m16n16k16` | A vector expression of eight `.s32` registers. |
| `.m8n32k16` |
| `.m32n8k16` |

Floating point fragments

| Data-type | Matrix | Fragment |
| --- | --- | --- |
| `.f16` | A or B | A vector expression of eight `.f16x2` registers. |
| `.f16` | C or D | A vector expression of four `.f16x2` registers. |
| `.f32` | A vector expression of eight `.f32` registers. |

Floating point fragments for `.bf16` data format

Multiplicands (A or B):

| Data-type | Shape | Matrix | Fragment |
| --- | --- | --- | --- |
| `.bf16` | `.m16n16k16` | A | A vector expression of four `.b32` registers, with each register containing two elements from the matrix. |
| B |
| `.m8n32k16` | A | A vector expression containing a two `.b32` registers, with containing two elements from the matrix. |
| B | A vector expression of eight `.b32` registers, with each register containing two elements from the matrix. |
| `.m32n8k16` | A | A vector expression of eight `.b32` registers, with each register containing two elements from the matrix. |
| B | A vector expression containing two `.b32` registers, with each containing two elements from the matrix. |

Accumulators (C or D):

| Data-type | Matrix | Fragment |
| --- | --- | --- |
| `.f32` | C or D | A vector expression containing eight `.f32` registers. |

Floating point fragments for `.tf32` data format

Multiplicands (A or B):

| Data-type | Shape | Matrix | Fragment |
| --- | --- | --- | --- |
| `.tf32` | `.m16n16k8` | A | A vector expression of four `.b32` registers. |
| B | A vector expression of four `.b32` registers. |

Accumulators (C or D):

| Data-type | Shape | Matrix | Fragment |
| --- | --- | --- | --- |
| `.f32` | `.m16n16k8` | C or D | A vector expression containing eight `.f32` registers. |

Double precision floating point fragments

Multiplicands (A or B):

| Data-type | Shape | Matrix | Fragment |
| --- | --- | --- | --- |
| `.f64` | `.m8n8k4` | A or B | A vector expression of single `.f64` register. |

Accumulators (C or D):

| Data-type | Shape | Matrix | Fragment |
| --- | --- | --- | --- |
| `.f64` | `.m8n8k4` | C or D | A vector expression containing single `.f64` register. |

Sub-byte integer and single-bit fragments

Multiplicands (A or B):

| Data-type | Shape | Fragment |
| --- | --- | --- |
| `.u4` or `.s4` | `.m8n8k32` | A vector expression containing a single `.b32` register, containing eight elements from the matrix. |
| `.b1` | `.m8n8k128` | A vector expression containing a single `.b32` register, containing 32 elements from the matrix. |

Accumulators (C or D):

| Data-type | Shape | Fragment |
| --- | --- | --- |
| `.s32` | `.m8n8k32` | A vector expression of two `.s32` registers. |
| `.m8n8k128` | A vector expression of two `.s32` registers. |

Manipulating fragment contents

The contents of a matrix fragment can be manipulated by reading and writing to individual
registers in the fragment, provided the following conditions are satisfied:

* All matrix element in the fragment are operated on uniformly across threads, using the same
  parameters.
* The order of the matrix elements is not changed.

For example, if each register corresponding to a given matrix is multiplied by a uniform constant
value, then the resulting matrix is simply the scaled version of the original matrix.

Note that type conversion between `.f16` and `.f32` accumulator fragments is not supported in
either direction. The result is undefined even if the order of elements in the fragment remains
unchanged.