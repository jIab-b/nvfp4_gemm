###### 9.7.14.6.2.2. Matrix Fragments for sparse mma.m16n8k32 with .f16 and .bf16 types 

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