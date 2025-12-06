###### 9.7.14.6.2.6. Matrix Fragments for sparse mma.m16n8k64 with .u8 / .s8 / .e4m3 / .e5m2 type 

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