##### 9.7.14.6.2. Matrix fragments for multiply-accumulate operation with sparse matrix A 

In this section we describe how the contents of thread registers are associated with fragments of
various matrices and the sparsity metadata. The following conventions are used throughout this
section:

* For matrix A, only the layout of a fragment is described in terms of register vector sizes and
  their association with the matrix data.
* For matrix B, when the combination of matrix dimension and the supported data type is not already
  covered in [Matrix multiply-accumulate operation using mma instruction](#warp-level-matrix-instructions-for-mma), a pictorial representation of matrix
  fragments is provided.
* For matrices C and D, since the matrix dimension - data type combination is the same for all
  supported shapes, and is already covered in
  [Matrix multiply-accumulate operation using mma instruction](#warp-level-matrix-instructions-for-mma), the pictorial representations
  of matrix fragments are not included in this section.
* For the metadata operand, pictorial representations of the association between indices of the
  elements of matrix A and the contents of the metadata operand are included. `Tk: [m..n]` present
  in cell `[x][y..z]` indicates that bits `m` through `n` (with `m` being higher) in the
  metadata operand of thread with `%laneid=k` contains the indices of the non-zero elements from
  the chunk `[x][y]..[x][z]` of matrix A.

###### 9.7.14.6.2.1. [Matrix Fragments for sparse `mma.m16n8k16` with `.f16` and `.bf16` types](#warp-level-matrix-fragment-sparse-mma-16816-f16bf16)[](#warp-level-matrix-fragment-sparse-mma-16816-f16bf16 "Permalink to this headline")

A warp executing sparse `mma.m16n8k16` with `.f16` / `.bf16` floating point type will compute
an MMA operation of shape `.m16n8k16`.

Elements of the matrix are distributed across the threads in a warp so each thread of the warp holds
a fragment of the matrix.

* Multiplicand A:

  | .atype | Fragment | Elements |
  | --- | --- | --- |
  | `.f16` / `.bf16` | A vector expression containing two `.b32` registers, with each register containing two non-zero `.f16` / `.bf16` elements out of 4 consecutive elements from matrix A. | Mapping of the non-zero elements is as described in [Sparse matrix storage](#warp-level-sparse-matrix-storage). |

  The layout of the fragments held by different threads is shown in [Figure 118](#sparse-mma-16816-f16-bf16-a).

  ![_images/sparse-mma-16816-f16-bf16-A.png](_images/sparse-mma-16816-f16-bf16-A.png)


  Figure 118 Sparse MMA .m16n8k16 fragment layout for matrix A with `.f16`/`.bf16` type.[](#sparse-mma-16816-f16-bf16-a "Permalink to this image")

  The row and column of a matrix fragment can be computed as:

  ```
  groupID = %laneid >> 2

  threadID_in_group = %laneid % 4



  row = groupID for a0 and a1

   groupID + 8 for a2 and a3



  col = [firstcol ... lastcol] // As per the mapping of non-zero elements

   // as described in Sparse matrix storage



  Where

  firstcol = threadID_in_group * 4

  lastcol = firstcol + 3
  ```
* Matrix fragments for multiplicand B and accumulators C and D are the same as in case of
  [Matrix Fragments for mma.m16n8k16 with floating point type](#warp-level-matrix-fragment-mma-16816-float) for `.f16`/`.b16` formats.
* Metadata: A `.b32` register containing 16 2-bit vectors each storing the index of a non-zero
  element of a 4-wide chunk of matrix A as shown in [Figure 119](#sparse-mma-metadata-16816-f16bf16).

  > ![_images/sparse-mma-metadata-16816-f16bf16.png](_images/sparse-mma-metadata-16816-f16bf16.png)
  >
  >
  > Figure 119 Sparse MMA .m16n8k16 metadata layout for `.f16`/`.bf16` type.[](#sparse-mma-metadata-16816-f16bf16 "Permalink to this image")

###### 9.7.14.6.2.2. [Matrix Fragments for sparse `mma.m16n8k32` with `.f16` and `.bf16` types](#warp-level-matrix-fragment-sparse-mma-16832-f16bf16)[](#warp-level-matrix-fragment-sparse-mma-16832-f16bf16 "Permalink to this headline")

A warp executing sparse `mma.m16n8k32` with `.f16` / `.bf16` floating point type will compute
an MMA operation of shape `.m16n8k32`.

Elements of the matrix are distributed across the threads in a warp so each thread of the warp holds
a fragment of the matrix.

* Multiplicand A:

  | .atype | Fragment | Elements |
  | --- | --- | --- |
  | `.f16` / `.bf16` | A vector expression containing four `.b32` registers, with each register containing two non-zero `.f16` / `.bf16` elements out of 4 consecutive elements from matrix A. | Mapping of the non-zero elements is as described in [Sparse matrix storage](#warp-level-sparse-matrix-storage). |

  The layout of the fragments held by different threads is shown in [Figure 120](#sparse-mma-16832-f16-bf16-a).

  ![_images/sparse-mma-16832-f16-bf16-A.png](_images/sparse-mma-16832-f16-bf16-A.png)


  Figure 120 Sparse MMA .m16n8k32 fragment layout for matrix A with `.f16`/`.bf16` type.[](#sparse-mma-16832-f16-bf16-a "Permalink to this image")

  The row and column of a matrix fragment can be computed as:

  ```
  groupID = %laneid >> 2

  threadID_in_group = %laneid % 4



  row = groupID for ai where 0 <= i < 2 || 4 <= i < 6

   groupID + 8 Otherwise



  col = [firstcol ... lastcol] // As per the mapping of non-zero elements

   // as described in Sparse matrix storage



  Where

  firstcol = threadID_in_group * 4 For ai where i < 4

   (threadID_in_group * 4) + 16 for ai where i >= 4

  lastcol = firstcol + 3
  ```
* Multiplicand B:

  | .atype | Fragment | Elements (low to high) |
  | --- | --- | --- |
  | `.f16` / `.bf16` | A vector expression containing four `.b32` registers, each containing two `.f16` / `.bf16` elements from matrix B. | b0, b1, b2, b3 |

  The layout of the fragments held by different threads is shown in [Figure 121](#sparse-mma-16832-f16bf16-b).

  ![_images/sparse-mma-16832-f16bf16-B.png](_images/sparse-mma-16832-f16bf16-B.png)


  Figure 121 Sparse MMA .m16n8k32 fragment layout for matrix B with `.f16`/`.bf16` type.[](#sparse-mma-16832-f16bf16-b "Permalink to this image")
* Matrix fragments for accumulators C and D are the same as in case of
  [Matrix Fragments for mma.m16n8k16 with floating point type](#warp-level-matrix-fragment-mma-16816-float)
  for `.f16`/`.b16` formats.
* Metadata: A `.b32` register containing 16 2-bit vectors with each pair of 2-bit vectors storing
  the indices of two non-zero element from a 4-wide chunk of matrix A as shown in
  [Figure 122](#sparse-mma-metadata-16832-f16bf16).

  > ![_images/sparse-mma-metadata-16832-f16bf16.png](_images/sparse-mma-metadata-16832-f16bf16.png)
  >
  >
  > Figure 122 Sparse MMA .m16n8k32 metadata layout for `.f16`/`.bf16` type.[](#sparse-mma-metadata-16832-f16bf16 "Permalink to this image")

###### 9.7.14.6.2.3. [Matrix Fragments for sparse `mma.m16n8k16` with `.tf32` floating point type](#warp-level-matrix-fragment-sparse-mma-16816-tf32)[](#warp-level-matrix-fragment-sparse-mma-16816-tf32 "Permalink to this headline")

A warp executing sparse `mma.m16n8k16` with `.tf32` floating point type will compute an MMA
operation of shape `.m16n8k16`.

Elements of the matrix are distributed across the threads in a warp so each thread of the warp holds
a fragment of the matrix.

* Multiplicand A:

  | .atype | Fragment | Elements |
  | --- | --- | --- |
  | `.tf32` | A vector expression containing four `.b32` registers, with each register containing one non-zero `.tf32` element out of 2 consecutive elements from matrix A. | Mapping of the non-zero elements is as described in [Sparse matrix storage](#warp-level-sparse-matrix-storage). |

  The layout of the fragments held by different threads is shown in [Figure 123](#sparse-mma-16816-tf32-a).

  ![_images/sparse-mma-16816-tf32-A.png](_images/sparse-mma-16816-tf32-A.png)


  Figure 123 Sparse MMA .m16n8k16 fragment layout for matrix A with `.tf32` type.[](#sparse-mma-16816-tf32-a "Permalink to this image")

  The row and column of a matrix fragment can be computed as:

  ```
  groupID = %laneid >> 2

  threadID_in_group = %laneid % 4



  row = groupID for a0 and a2

   groupID + 8 for a1 and a3



  col = [firstcol ... lastcol] // As per the mapping of non-zero elements

   // as described in Sparse matrix storage



  Where

  firstcol = threadID_in_group * 2 for a0 and a1

   (threadID_in_group * 2) + 8 for a2 and a3

  lastcol = firstcol + 1
  ```
* Multiplicand B:

  | .atype | Fragment | Elements (low to high) |
  | --- | --- | --- |
  | `.tf32` | A vector expression containing four `.b32` registers, each containing four `.tf32` elements from matrix B. | b0, b1, b2, b3 |

  The layout of the fragments held by different threads is shown in [Figure 124](#sparse-mma-16816-tf32-b).

  ![_images/sparse-mma-16816-tf32-B.png](_images/sparse-mma-16816-tf32-B.png)


  Figure 124 Sparse MMA .m16n8k16 fragment layout for matrix B with `.tf32` type.[](#sparse-mma-16816-tf32-b "Permalink to this image")
* Matrix fragments for accumulators C and D are the same as in case of
  [Matrix Fragments for mma.m16n8k16 with floating point type](#warp-level-matrix-fragment-mma-16816-float).
* Metadata: A `.b32` register containing 8 4-bit vectors each storing the index of a non-zero
  element of a 2-wide chunk of matrix A as shown in [Figure 125](#sparse-mma-metadata-16816-tf32).

  > ![_images/sparse-mma-metadata-16816-tf32.png](_images/sparse-mma-metadata-16816-tf32.png)
  >
  >
  > Figure 125 Sparse MMA .m16n8k16 metadata layout for `.tf32` type.[](#sparse-mma-metadata-16816-tf32 "Permalink to this image")

###### 9.7.14.6.2.4. [Matrix Fragments for sparse `mma.m16n8k8` with `.tf32` floating point type](#warp-level-matrix-fragment-sparse-mma-1688-tf32)[](#warp-level-matrix-fragment-sparse-mma-1688-tf32 "Permalink to this headline")

A warp executing sparse `mma.m16n8k8` with `.tf32` floating point type will compute an MMA
operation of shape `.m16n8k8`.

Elements of the matrix are distributed across the threads in a warp so each thread of the warp holds
a fragment of the matrix.

* Multiplicand A:

  | .atype | Fragment | Elements |
  | --- | --- | --- |
  | `.tf32` | A vector expression containing two `.b32` registers, each containing one non-zero `.tf32` element out of 2 consecutive elements from matrix A. | Mapping of the non-zero elements is as described in [Sparse matrix storage](#warp-level-sparse-matrix-storage). |

  The layout of the fragments held by different threads is shown in [Figure 126](#sparse-mma-1688-tf32).

  ![_images/sparse-mma-1688-tf32-A.png](_images/sparse-mma-1688-tf32-A.png)


  Figure 126 Sparse MMA .m16n8k8 fragment layout for matrix A with `.tf32` type.[](#sparse-mma-1688-tf32 "Permalink to this image")

  The row and column of a matrix fragment can be computed as:

  ```
  groupID = %laneid >> 2

  threadID_in_group = %laneid % 4



  row = groupID for a0

   groupID + 8 for a1



  col = [firstcol ... lastcol] // As per the mapping of non-zero elements

   // as described in Sparse matrix storage



  Where

  firstcol = threadID_in_group * 2

  lastcol = firstcol + 1
  ```
* Matrix fragments for multiplicand B and accumulators C and D are the same as in case of
  [Matrix Fragments for mma.m16n8k8](#warp-level-matrix-fragment-mma-1688) for `.tf32`
  format.
* Metadata: A `.b32` register containing 8 4-bit vectors each storing the index of a non-zero
  element of a 2-wide chunk of matrix A as shown in [Figure 127](#sparse-mma-metadata-1688-tf32).

  > ![_images/sparse-mma-metadata-1688-tf32.png](_images/sparse-mma-metadata-1688-tf32.png)
  >
  >
  > Figure 127 Sparse MMA .m16n8k8 metadata layout for `.tf32` type.[](#sparse-mma-metadata-1688-tf32 "Permalink to this image")

###### 9.7.14.6.2.5. [Matrix Fragments for sparse `mma.m16n8k32` with `.u8` / `.s8` integer type](#warp-level-matrix-fragment-sparse-mma-16832-u8s8)[](#warp-level-matrix-fragment-sparse-mma-16832-u8s8 "Permalink to this headline")

A warp executing sparse `mma.m16n8k32` with `.u8` / `.s8` integer type will compute an MMA
operation of shape `.m16n8k32`.

Elements of the matrix are distributed across the threads in a warp so each thread of the warp holds
a fragment of the matrix.

* Multiplicand A:

  | .atype | Fragment | Elements |
  | --- | --- | --- |
  | `.u8` / `.s8` | A vector expression containing two `.b32` registers, with each register containing four non-zero `.u8` / `.s8` elements out of 8 consecutive elements from matrix A. | Mapping of the non-zero elements is as described in [Sparse matrix storage](#warp-level-sparse-matrix-storage). |

  The layout of the fragments held by different threads is shown in [Figure 128](#sparse-mma-16832-u8s8-a).

  ![_images/sparse-mma-16832-u8s8-A.png](_images/sparse-mma-16832-u8s8-A.png)


  Figure 128 Sparse MMA .m16n8k32 fragment layout for matrix A with `.u8`/`.s8` type.[](#sparse-mma-16832-u8s8-a "Permalink to this image")

  ```
  groupID = %laneid >> 2

  threadID_in_group = %laneid % 4



  row = groupID for ai where 0 <= i < 4

   groupID + 8 Otherwise



  col = [firstcol ... lastcol] // As per the mapping of non-zero elements

   // as described in Sparse matrix storage



  Where

  firstcol = threadID_in_group * 8

  lastcol = firstcol + 7
  ```
* Matrix fragments for multiplicand B and accumulators C and D are the same as in case of
  [Matrix Fragments for mma.m16n8k32](#warp-level-matrix-fragment-mma-16832).
* Metadata: A `.b32` register containing 16 2-bit vectors with each pair of 2-bit vectors storing
  the indices of two non-zero elements from a 4-wide chunk of matrix A as shown in
  [Figure 129](#sparse-mma-metadata-16832-u8s8).

  > ![_images/sparse-mma-metadata-16832-u8s8.png](_images/sparse-mma-metadata-16832-u8s8.png)
  >
  >
  > Figure 129 Sparse MMA .m16n8k32 metadata layout for `.u8`/`.s8` type.[](#sparse-mma-metadata-16832-u8s8 "Permalink to this image")

###### 9.7.14.6.2.6. [Matrix Fragments for sparse `mma.m16n8k64` with `.u8` / `.s8` / `.e4m3` / `.e5m2` type](#warp-level-matrix-fragment-sparse-mma-16864-u8s8-fp8)[](#warp-level-matrix-fragment-sparse-mma-16864-u8s8-fp8 "Permalink to this headline")

A warp executing sparse `mma.m16n8k64` with `.u8` / `.s8`/ `.e4m3`/ `.e5m2` /
`.e3m2` / `.e2m3` / `.e2m1` type will compute an MMA operation of shape `.m16n8k64`.

Elements of the matrix are distributed across the threads in a warp so each thread of the warp holds
a fragment of the matrix.

* Multiplicand A:

  | .atype | Fragment | Elements |
  | --- | --- | --- |
  | `.u8` / `.s8` | A vector expression containing four `.b32` registers, with each register containing four non-zero `.u8` / `.s8` elements out of 8 consecutive elements from matrix A. | Mapping of the non-zero elements is as described in [Sparse matrix storage](#warp-level-sparse-matrix-storage). |
  | `.e4m3` / `.e5m2` / `.e3m2` / `.e2m3` / `.e2m1` | A vector expression containing four `.b32` registers, with each register containing four non-zero `.e4m3` / `.e5m2` / `.e3m2` / `.e2m3` / `.e2m1` elements out of 8 consecutive elements from matrix A. |

  The layout of the fragments held by different threads is shown in [Figure 130](#sparse-mma-16864-u8s8-a-first32col)
  and [Figure 131](#sparse-mma-16864-u8s8-a-last32col).

  ![_images/sparse-mma-16864-u8s8-A-first32col.png](_images/sparse-mma-16864-u8s8-A-first32col.png)


  Figure 130 Sparse MMA .m16n8k64 fragment layout for columns 0–31 of matrix A with `.u8`/`.s8`/`.e4m3`/`.e5m2`/`.e3m2`/`.e2m3`/`.e2m1` type.[](#sparse-mma-16864-u8s8-a-first32col "Permalink to this image")


  ![_images/sparse-mma-16864-u8s8-A-last32col.png](_images/sparse-mma-16864-u8s8-A-last32col.png)


  Figure 131 Sparse MMA .m16n8k64 fragment layout for columns 32–63 of matrix A with `.u8`/`.s8`/`.e4m3`/`.e5m2`/`.e3m2`/`.e2m3`/`.e2m1` type.[](#sparse-mma-16864-u8s8-a-last32col "Permalink to this image")

  ```
  groupID = %laneid >> 2

  threadID_in_group = %laneid % 4



  row = groupID for ai where 0 <= i < 4 || 8 <= i < 12

   groupID + 8 Otherwise



  col = [firstcol ... lastcol] // As per the mapping of non-zero elements

   // as described in Sparse matrix storage



  Where

  firstcol = threadID_in_group * 8 For ai where i < 8

   (threadID_in_group * 8) + 32 For ai where i >= 8

  lastcol = firstcol + 7
  ```
* Multiplicand B:

  | .btype | Fragment | Elements (low to high) |
  | --- | --- | --- |
  | `.u8` / `.s8` | A vector expression containing four `.b32` registers, each containing four `.u8` / `.s8` elements from matrix B. | b0, b1, b2, b3, …, b15 |
  | `.e4m3` / `.e5m2` / `.e3m2` / `.e2m3` / `.e2m1` | A vector expression containing four `.b32` registers, each containing four `.e4m3` / `.e5m2` / `.e3m2` / `.e2m3` / `.e2m1` elements from matrix B. |

  The layout of the fragments held by different threads is shown in [Figure 132](#sparse-mma-16864-u8s8-b1),
  [Figure 133](#sparse-mma-16864-u8s8-b2), [Figure 134](#sparse-mma-16864-u8s8-b3) and [Figure 135](#sparse-mma-16864-u8s8-b4).

  ![_images/sparse-mma-16864-u8s8-B1.png](_images/sparse-mma-16864-u8s8-B1.png)


  Figure 132 Sparse MMA .m16n8k64 fragment layout for rows 0–15 of matrix B with `.u8`/`.s8`/`.e4m3`/`.e5m2`/`.e3m2`/`.e2m3`/`.e2m1` type.[](#sparse-mma-16864-u8s8-b1 "Permalink to this image")


  ![_images/sparse-mma-16864-u8s8-B2.png](_images/sparse-mma-16864-u8s8-B2.png)


  Figure 133 Sparse MMA .m16n8k64 fragment layout for rows 16–31 of matrix B with `.u8`/`.s8`/`.e4m3`/`.e5m2`/`.e3m2`/`.e2m3`/`.e2m1` type.[](#sparse-mma-16864-u8s8-b2 "Permalink to this image")


  ![_images/sparse-mma-16864-u8s8-B3.png](_images/sparse-mma-16864-u8s8-B3.png)


  Figure 134 Sparse MMA .m16n8k64 fragment layout for rows 32–47 of matrix B with `.u8`/`.s8`/`.e4m3`/`.e5m2`/`.e3m2`/`.e2m3`/`.e2m1` type.[](#sparse-mma-16864-u8s8-b3 "Permalink to this image")


  ![_images/sparse-mma-16864-u8s8-B4.png](_images/sparse-mma-16864-u8s8-B4.png)


  Figure 135 Sparse MMA .m16n8k64 fragment layout for rows 48–63 of matrix B with `.u8`/`.s8`/`.e4m3`/`.e5m2`/`.e3m2`/`.e2m3`/`.e2m1` type.[](#sparse-mma-16864-u8s8-b4 "Permalink to this image")
* Matrix fragments for accumulators C and D are the same as in case of
  [Matrix Fragments for mma.m16n8k16 with integer type](#warp-level-matrix-fragment-mma-16816-i8-f8).
* Metadata: A `.b32` register containing 16 2-bit vectors with each pair of 2-bit vectors storing
  the indices of two non-zero elements from a 4-wide chunk of matrix A as shown in
  [Figure 136](#sparse-mma-metadata-16864-u8s8-first32col) and [Figure 137](#sparse-mma-metadata-16864-u8s8-last32col).

  > ![_images/sparse-mma-metadata-16864-u8s8-first32col.png](_images/sparse-mma-metadata-16864-u8s8-first32col.png)
  >
  >
  > Figure 136 Sparse MMA .m16n8k64 metadata layout for columns 0–31 for `.u8`/`.s8`/`.e4m3`/`.e5m2`/`.e3m2`/`.e2m3`/`.e2m1` type.[](#sparse-mma-metadata-16864-u8s8-first32col "Permalink to this image")
  >
  >
  > ![_images/sparse-mma-metadata-16864-u8s8-last32col.png](_images/sparse-mma-metadata-16864-u8s8-last32col.png)
  >
  >
  > Figure 137 Sparse MMA .m16n8k64 metadata layout for columns 32–63 for `.u8`/`.s8`/`.e4m3`/`.e5m2`/`.e3m2`/`.e2m3`/`.e2m1` type.[](#sparse-mma-metadata-16864-u8s8-last32col "Permalink to this image")

###### 9.7.14.6.2.7. [Matrix Fragments for sparse `mma.m16n8k64` with `.u4` / `.s4` integer type](#warp-level-matrix-fragment-sparse-mma-16864-u4s4)[](#warp-level-matrix-fragment-sparse-mma-16864-u4s4 "Permalink to this headline")

A warp executing sparse `mma.m16n8k64` with `.u4` / `.s4` integer type will compute an MMA
operation of shape `.m16n8k64`.

Elements of the matrix are distributed across the threads in a warp so each thread of the warp holds
a fragment of the matrix.

* Multiplicand A:

  | .atype | Fragment | Elements |
  | --- | --- | --- |
  | `.u4` / `.s4` | A vector expression containing two `.b32` registers, with each register containing eight non-zero `.u4` / `.s4` elements out of 16 consecutive elements from matrix A. | Mapping of the non-zero elements is as described in [Sparse matrix storage](#warp-level-sparse-matrix-storage). |

  The layout of the fragments held by different threads is shown in [Figure 138](#sparse-mma-16864-u4s4-a).

  ![_images/sparse-mma-16864-u4s4-A.png](_images/sparse-mma-16864-u4s4-A.png)


  Figure 138 Sparse MMA .m16n8k64 fragment layout for matrix A with `.u4`/`.s4` type.[](#sparse-mma-16864-u4s4-a "Permalink to this image")

  ```
  groupID = %laneid >> 2

  threadID_in_group = %laneid % 4



  row = groupID for ai where 0 <= i < 8

   groupID + 8 Otherwise



  col = [firstcol ... lastcol] // As per the mapping of non-zero elements

   // as described in Sparse matrix storage



  Where

  firstcol = threadID_in_group * 16

  lastcol = firstcol + 15
  ```
* Matrix fragments for multiplicand B and accumulators C and D are the same as in case of
  [Matrix Fragments for mma.m16n8k64](#warp-level-matrix-fragment-mma-16864).
* Metadata: A `.b32` register containing 16 2-bit vectors with each pair of 2-bit vectors storing
  the indices of four non-zero elements from a 8-wide chunk of matrix A as shown in
  [Figure 139](#sparse-mma-metadata-16864-u4s4).

  > ![_images/sparse-mma-metadata-16864-u4s4.png](_images/sparse-mma-metadata-16864-u4s4.png)
  >
  >
  > Figure 139 Sparse MMA .m16n8k64 metadata layout for `.u4`/`.s4` type.[](#sparse-mma-metadata-16864-u4s4 "Permalink to this image")

###### 9.7.14.6.2.8. [Matrix Fragments for sparse `mma.m16n8k128` with `.u4` / `.s4` integer type](#warp-level-matrix-fragment-sparse-mma-168128-u4s4)[](#warp-level-matrix-fragment-sparse-mma-168128-u4s4 "Permalink to this headline")

A warp executing sparse `mma.m16n8k128` with `.u4` / `.s4` / `.e2m1` integer type will compute an MMA
operation of shape `.m16n8k128`.

Elements of the matrix are distributed across the threads in a warp so each thread of the warp holds
a fragment of the matrix.

* Multiplicand A:

  | .atype | Fragment | Elements |
  | --- | --- | --- |
  | `.u4` / `.s4` | A vector expression containing four `.b32` registers, with each register containing eight non-zero `.u4` / `.s4` elements out of 16 consecutive elements from matrix A. | Mapping of the non-zero elements is as described in [Sparse matrix storage](#warp-level-sparse-matrix-storage). |
  | `.e2m1` | A vector expression containing four `.b32` registers, with each register containing eight non-zero `.e2m1` elements out of 16 consecutive elements from matrix A. |

  The layout of the fragments held by different threads is shown in [Figure 140](#sparse-mma-168128-u4s4-a-first64col)
  and [Figure 141](#sparse-mma-168128-u4s4-a-last64col).

  ![_images/sparse-mma-168128-u4s4-A-first64col.png](_images/sparse-mma-168128-u4s4-A-first64col.png)


  Figure 140 Sparse MMA .m16n8k128 fragment layout for columns 0–63 of matrix A with `.u4`/`.s4`/`.e2m1` type.[](#sparse-mma-168128-u4s4-a-first64col "Permalink to this image")


  ![_images/sparse-mma-168128-u4s4-A-last64col.png](_images/sparse-mma-168128-u4s4-A-last64col.png)


  Figure 141 Sparse MMA .m16n8k128 fragment layout for columns 64–127 of matrix A with `.u4`/`.s4`/`.e2m1` type.[](#sparse-mma-168128-u4s4-a-last64col "Permalink to this image")

  ```
  groupID = %laneid >> 2

  threadID_in_group = %laneid % 4



  row = groupID for ai where 0 <= i < 8 || 16 <= i < 24

   groupID + 8 Otherwise



  col = [firstcol ... lastcol] // As per the mapping of non-zero elements

   // as described in Sparse matrix storage



  Where

  firstcol = threadID_in_group * 16 For ai where i < 16

   (threadID_in_group * 16) + 64 For ai where i >= 16

  lastcol = firstcol + 15
  ```
* Multiplicand B:

  | .atype | Fragment | Elements (low to high) |
  | --- | --- | --- |
  | `.u4` / `.s4` | A vector expression containing four `.b32` registers, each containing eight `.u4` / `.s4` elements from matrix B. | b0, b1, b2, b3, …, b31 |
  | `.e2m1` | A vector expression containing four `.b32` registers, each containing eight `.e2m1` elements from matrix B. |

  The layout of the fragments held by different threads is shown in [Figure 142](#sparse-mma-168128-u4s4-b1),
  [Figure 143](#sparse-mma-168128-u4s4-b2), [Figure 144](#sparse-mma-168128-u4s4-b3), [Figure 145](#sparse-mma-168128-u4s4-b4).

  ![_images/sparse-mma-168128-u4s4-B1.png](_images/sparse-mma-168128-u4s4-B1.png)


  Figure 142 Sparse MMA .m16n8k128 fragment layout for rows 0–31 of matrix B with `.u4`/`.s4`/`.e2m1` type.[](#sparse-mma-168128-u4s4-b1 "Permalink to this image")


  ![_images/sparse-mma-168128-u4s4-B2.png](_images/sparse-mma-168128-u4s4-B2.png)


  Figure 143 Sparse MMA .m16n8k128 fragment layout for rows 32–63 of matrix B with `.u4`/`.s4`/`.e2m1` type.[](#sparse-mma-168128-u4s4-b2 "Permalink to this image")


  ![_images/sparse-mma-168128-u4s4-B3.png](_images/sparse-mma-168128-u4s4-B3.png)


  Figure 144 Sparse MMA .m16n8k128 fragment layout for rows 64–95 of matrix B with `.u4`/`.s4`/`.e2m1` type.[](#sparse-mma-168128-u4s4-b3 "Permalink to this image")


  ![_images/sparse-mma-168128-u4s4-B4.png](_images/sparse-mma-168128-u4s4-B4.png)


  Figure 145 Sparse MMA .m16n8k128 fragment layout for rows 96–127 of matrix B with `.u4`/`.s4`/`.e2m1` type.[](#sparse-mma-168128-u4s4-b4 "Permalink to this image")
* Matrix fragments for accumulators C and D are the same as in case of
  [Matrix Fragments for mma.m16n8k64](#warp-level-matrix-fragment-mma-16864).
* Metadata: A `.b32` register containing 16 2-bit vectors with each pair of 2-bit vectors storing
  the indices of four non-zero elements from a 8-wide chunk of matrix A as shown in
  [Figure 146](#sparse-mma-metadata-168128-u4s4-first64col) and [Figure 147](#sparse-mma-metadata-168128-u4s4-last64col).

  > ![_images/sparse-mma-metadata-168128-u4s4-first64col.png](_images/sparse-mma-metadata-168128-u4s4-first64col.png)
  >
  >
  > Figure 146 Sparse MMA .m16n8k128 metadata layout for columns 0–63 for `.u4`/`.s4`/`.e2m1` type.[](#sparse-mma-metadata-168128-u4s4-first64col "Permalink to this image")
  >
  >
  > ![_images/sparse-mma-metadata-168128-u4s4-last64col.png](_images/sparse-mma-metadata-168128-u4s4-last64col.png)
  >
  >
  > Figure 147 Sparse MMA .m16n8k128 metadata layout for columns 64–127 for `.u4`/`.s4`/`.e2m1` type.[](#sparse-mma-metadata-168128-u4s4-last64col "Permalink to this image")