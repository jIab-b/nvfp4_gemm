##### 9.7.14.5.12. Matrix Fragments for mma.m16n8k128 

A warp executing `mma.m16n8k128` will compute an MMA operation of shape `.m16n8k128`.

Elements of the matrix are distributed across the threads in a warp so each thread of the warp holds
a fragment of the matrix.

* Multiplicand A:

  | .atype | Fragment | Elements (low to high) |
  | --- | --- | --- |
  | `.b1` | A vector expression containing two `.b32` registers, with each register containing thirty two `.b1` elements from the matrix A. | a0, a1, …, a62, a63 |

  The layout of the fragments held by different threads is shown in [Figure 97](#mma-168128-a).

  ![_images/mma-168128-A.png](_images/mma-168128-A.png)


  Figure 97 MMA .m16n8k128 fragment layout for matrix A with `.b1` type.[](#mma-168128-a "Permalink to this image")

  The row and column of a matrix fragment can be computed as:

  ```
  groupID = %laneid >> 2

  threadID_in_group = %laneid % 4



  row = groupID for ai where i < 32

   groupID + 8 for ai where i >= 32



  col = (threadID_in_group * 32) + (i & 0x1F) for ai where i = {0, ...,63}
  ```
* Multiplicand B:

  | .btype | Fragment | Elements (low to high) |
  | --- | --- | --- |
  | `.b1` | A vector expression containing a single `.b32` register containing thirty two `.b1` elements from the matrix B. | b0, b1, … , b30, b31 |

  The layout of the fragments held by different threads is shown in [Figure 98](#mma-168128-b).

  ![_images/mma-168128-B.png](_images/mma-168128-B.png)


  Figure 98 MMA .m16n8k128 fragment layout for matrix B with `.b1` type.[](#mma-168128-b "Permalink to this image")

  The row and column of a matrix fragment can be computed as:

  ```
  groupID = %laneid >> 2

  threadID_in_group = %laneid % 4



  row = (threadID_in_group * 32) + i for bi where i = {0,...,31}

  col = groupID
  ```
* Accumulators (C or D):

  | .ctype / .dtype | Fragment | Elements (low to high) |
  | --- | --- | --- |
  | `.s32` | A vector expression containing four `.s32` registers, containing four `.s32` elements from the matrix C (or D). | c0, c1, c2, c3 |

  The layout of the fragments held by different threads is shown in [Figure 99](#mma-168128-c).

  ![_images/mma-168128-C.png](_images/mma-168128-C.png)


  Figure 99 MMA .m16n8k128 fragment layout for accumulator matrix C/D with `.s32` type.[](#mma-168128-c "Permalink to this image")

  The row and column of a matrix fragment can be computed as:

  ```
  groupID = %laneid >> 2

  threadID_in_group = %laneid % 4



  row = groupID for ci where i < 2

   groupID + 8 for ci where i >= 2



  col = (threadID_in_group * 2) + (i & 0x1) for ci where i = {0, 1, 2, 3}
  ```