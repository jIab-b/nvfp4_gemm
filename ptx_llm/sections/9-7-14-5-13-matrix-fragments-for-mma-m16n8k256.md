##### 9.7.14.5.13. Matrix Fragments for mma.m16n8k256 

A warp executing `mma.m16n8k256` will compute an MMA operation of shape `.m16n8k256`.

Elements of the matrix are distributed across the threads in a warp so each thread of the warp holds
a fragment of the matrix.

* Multiplicand A:

  | .atype | Fragment | Elements (low to high) |
  | --- | --- | --- |
  | `.b1` | A vector expression containing four `.b32` registers, with each register containing thirty two `.b1` elements from the matrix A. | a0, a1, …, a126, a127 |

  The layout of the fragments held by different threads is shown in [Figure 100](#mma-168256-a).

  ![_images/mma-168256-A.png](_images/mma-168256-A.png)


  Figure 100 MMA .m16n8k256 fragment layout for matrix A with `.b1` type.[](#mma-168256-a "Permalink to this image")

  The row and column of a matrix fragment can be computed as:

  ```
  groupID = %laneid >> 2

  threadID_in_group = %laneid % 4



  row = groupID for ai where 0 <= i < 32 || 64 <= i < 96

   groupID + 8 otherwise



  col = (threadID_in_group * 32) + i for ai where i < 64

   (threadID_in_group * 32) + (i & 0x1F) + 128 for ai where i >= 64
  ```
* Multiplicand B:

  | .btype | Fragment | Elements (low to high) |
  | --- | --- | --- |
  | `.b1` | A vector expression containing two `.b32` registers, with each register containing thirty two `.b1` elements from the matrix B. | b0, b1, …, b62, b63 |

  The layout of the fragments held by different threads is shown in [Figure 101](#mma-168256-b-1) and
  [Figure 102](#mma-168256-b-2).

  ![_images/mma-168256-B_1.png](_images/mma-168256-B_1.png)


  Figure 101 MMA .m16n8k256 fragment layout for rows 0–127 of matrix B with `.b1` type.[](#mma-168256-b-1 "Permalink to this image")


  ![_images/mma-168256-B_2.png](_images/mma-168256-B_2.png)


  Figure 102 MMA .m16n8k256 fragment layout for rows 128–255 of matrix B with `.b1` type.[](#mma-168256-b-2 "Permalink to this image")

  The row and column of a matrix fragment can be computed as:

  ```
  groupID = %laneid >> 2

  threadID_in_group = %laneid % 4



  row = (threadID_in_group * 32) + (i & 0x1F) for bi where i < 32

   (threadID_in_group * 32) + (i & 0x1F) + 128 for bi where i >= 32



  col = groupID
  ```
* Accumulators (C or D):

  | .ctype / .dtype | Fragment | Elements (low to high) |
  | --- | --- | --- |
  | `.s32` | A vector expression containing four `.s32` registers, containing four `.s32` elements from the matrix C (or D). | c0, c1, c2, c3 |

  The layout of the fragments held by different threads is shown in [Figure 103](#mma-168256-c).

  ![_images/mma-168256-C.png](_images/mma-168256-C.png)


  Figure 103 MMA .m16n8k256 fragment layout for accumulator matrix C/D with `.s32` type.[](#mma-168256-c "Permalink to this image")

  The row and column of a matrix fragment can be computed as:

  ```
  groupID = %laneid >> 2

  threadID_in_group = %laneid % 4



  row = groupID for ci where i < 2

   groupID + 8 for ci where i >= 2



  col = (threadID_in_group * 2) + (i & 0x1) for ci where i = {0, 1, 2, 3}
  ```