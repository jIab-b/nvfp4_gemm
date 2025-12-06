##### 9.7.14.5.11. Matrix Fragments for mma.m16n8k64 

A warp executing `mma.m16n8k64` will compute an MMA operation of shape `.m16n8k64`.

Elements of the matrix are distributed across the threads in a warp so each thread of the warp holds
a fragment of the matrix.

* Multiplicand A:

  | .atype | Fragment | Elements (low to high) |
  | --- | --- | --- |
  | `.s4` / `.u4` | A vector expression containing four `.b32` registers, with each register containing eight `.s4` / `.u4` elements from the matrix A. | a0, a1, …, a30, a31 |
  | `.e2m1` | A vector expression containing four `.b32` registers, with each register containing eight `.e2m1` elements from the matrix A. | a0, a1, …, a30, a31 |

  The layout of the fragments held by different threads is shown in [Figure 93](#mma-16864-a).

  ![_images/mma-16864-A.png](_images/mma-16864-A.png)


  Figure 93 MMA .m16n8k64 fragment layout for matrix A with `.u4` / `.s4` / `.e2m1` type.[](#mma-16864-a "Permalink to this image")

  The row and column of a matrix fragment can be computed as:

  ```
  groupID = %laneid >> 2

  threadID_in_group = %laneid % 4



  row = groupID for ai where 0 <= i < 8 || 16 <= i < 24

   groupID + 8 otherwise



  col = (threadID_in_group * 8) + (i & 0x7) for ai where i < 16

   (threadID_in_group * 8) + (i & 0x7) + 32 for ai where i >= 16
  ```
* Multiplicand B:

  | .btype | Fragment | Elements (low to high) |
  | --- | --- | --- |
  | `.s4` / `.u4` | A vector expression containing two `.b32` registers, with each register containing eight `.s4` / `.u4` elements from the matrix B. | b0, b1, …, b14, b15 |
  | `.e2m1` | A vector expression containing two `.b32` registers, with each register containing eight `.e2m1` elements from the matrix B. | b0, b1, …, b14, b15 |

  The layout of the fragments held by different threads is shown in [Figure 94](#mma-16864-b-1)
  and [Figure 95](#mma-16864-b-2).

  ![_images/mma-16864-B_1.png](_images/mma-16864-B_1.png)


  Figure 94 MMA .m16n8k64 fragment layout for rows 0–31 of matrix B with `.u4` / `.s4` / `.e2m1` type.[](#mma-16864-b-1 "Permalink to this image")


  ![_images/mma-16864-B_2.png](_images/mma-16864-B_2.png)


  Figure 95 MMA .m16n8k64 fragment layout for rows 32–63 of matrix B with `.u4` / `.s4` / `.e2m1` type.[](#mma-16864-b-2 "Permalink to this image")

  The row and column of a matrix fragment can be computed as:

  ```
  groupID = %laneid >> 2

  threadID_in_group = %laneid % 4



  row = (threadID_in_group * 8) + (i & 0x7) for bi where i < 8

   (threadID_in_group * 8) + (i & 0x7) + 32 for bi where i >= 8



  col = groupID
  ```
* Accumulators (C or D):

  | .ctype / .dtype | Fragment | Elements (low to high) |
  | --- | --- | --- |
  | `.s32` | A vector expression containing four `.s32` registers, containing four `.s32` elements from the matrix C (or D). | c0, c1, c2, c3 |
  | `.f32` | A vector expression containing four `.f32` registers, containing four `.f32` elements from the matrix C (or D). | c0, c1, c2, c3 |

  The layout of the fragments held by different threads is shown in [Figure 96](#mma-16864-c).

  ![_images/mma-16864-C.png](_images/mma-16864-C.png)


  Figure 96 MMA .m16n8k64 fragment layout for accumulator matrix C/D with `.s32` / `.f32` type.[](#mma-16864-c "Permalink to this image")

  The row and column of a matrix fragment can be computed as:

  ```
  groupID = %laneid >> 2

  threadID_in_group = %laneid % 4



  row = groupID for ci where i < 2

   groupID + 8 for ci where i >= 2



  col = (threadID_in_group * 2) + (i & 0x1) for ci where i = {0,..,3}
  ```