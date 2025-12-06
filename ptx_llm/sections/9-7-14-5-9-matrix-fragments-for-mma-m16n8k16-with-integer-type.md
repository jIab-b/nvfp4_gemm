##### 9.7.14.5.9. Matrix Fragments for mma.m16n8k16 with integer type 

A warp executing `mma.m16n8k16` will compute an MMA operation of shape `.m16n8k16`.

Elements of the matrix are distributed across the threads in a warp so each thread of the warp holds
a fragment of the matrix.

* Multiplicand A:

  | .atype | Fragment | Elements (low to high) |
  | --- | --- | --- |
  | `.u8` / `.s8` | A vector expression containing two `.b32` registers, with each register containing four `.u8` / `.s8` elements from the matrix A. | a0, a1, a2, a3, a4, a5, a6, a7 |
  | `.e4m3` / `.e5m2` | A vector expression containing two `.b32` registers, with each register containing four `.e4m3` / `.e5m2` elements from the matrix A. | a0, a1, a2, a3, a4, a5, a6, a7 |

  The layout of the fragments held by different threads is shown in [Figure 84](#mma-16816-a-i8).

  ![_images/mma-16816-A-i8.png](_images/mma-16816-A-i8.png)


  Figure 84 MMA .m16n8k16 fragment layout for matrix A with `.u8` / `.s8` type.[](#mma-16816-a-i8 "Permalink to this image")

  The row and column of a matrix fragment can be computed as:

  ```
  groupID = %laneid >> 2

  threadID_in_group = %laneid % 4



  row = groupID for ai where i < 4

   groupID + 8 for ai where i >= 4



  col = (threadID_in_group * 4) + (i & 0x3) for ai where i = {0,..,7}
  ```
* Multiplicand B:

  | .btype | Fragment | Elements (low to high) |
  | --- | --- | --- |
  | `.u8` / `.s8` | A vector expression containing a single `.b32` register, containing four `.u8` / `.s8` elements from the matrix B. | b0, b1, b2, b3 |
  | `.e4m3` / `.e5m2` | A vector expression containing a single `.b32` register, containing four `.e4m3` / `.e5m2` elements from the matrix B. | b0, b1. b2. b3 |

  The layout of the fragments held by different threads is shown in [Figure 85](#mma-16816-b-i8).

  ![_images/mma-16816-B-i8.png](_images/mma-16816-B-i8.png)


  Figure 85 MMA .m16n8k16 fragment layout for matrix B with `.u8` / `.s8` type.[](#mma-16816-b-i8 "Permalink to this image")

  The row and column of a matrix fragment can be computed as:

  ```
  groupID = %laneid >> 2

  threadID_in_group = %laneid % 4



  row = (threadID_in_group * 4) + i for bi where i = {0,..,3}



  col = groupID
  ```
* Accumulators (C or D):

  | .ctype / .dtype | Fragment | Elements (low to high) |
  | --- | --- | --- |
  | `.s32` | A vector expression containing four `.s32` registers, containing four `.s32` elements from the matrix C (or D). | c0, c1, c2, c3 |
  | `.f32` | A vector expression containing four `.f32` registers, containing four `.f32` elements from the matrix C (or D). | c0, c1, c2, c3 |
  | `.f16` | A vector expression containing two `.f16x2` registers, with each register containing two `.f16` elements from the matrix C (or D). | c0, c1, c1, c2 |

  The layout of the fragments held by different threads is shown in [Figure 86](#mma-16816-c-i8).

  ![_images/mma-16816-C-i8.png](_images/mma-16816-C-i8.png)


  Figure 86 MMA .m16n8k16 fragment layout for accumulator matrix C/D with `.s32` type.[](#mma-16816-c-i8 "Permalink to this image")

  The row and column of a matrix fragment can be computed as:

  ```
  groupID = %laneid >> 2

  threadID_in_group = %laneid % 4



  row = groupID for ci where i < 2

   groupID + 8 for ci where i >= 2



  col = (threadID_in_group * 2) + (i & 0x1) for ci where i = {0,..,3}
  ```