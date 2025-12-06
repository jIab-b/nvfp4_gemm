###### 9.7.14.6.2.7. Matrix Fragments for sparse mma.m16n8k64 with .u4 / .s4 integer type 

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