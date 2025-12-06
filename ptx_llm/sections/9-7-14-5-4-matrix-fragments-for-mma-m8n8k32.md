##### 9.7.14.5.4. Matrix Fragments for mma.m8n8k32 

A warp executing `mma.m8n8k32` will compute an MMA operation of shape `.m8n8k32`.

Elements of the matrix are distributed across the threads in a warp so each thread of the warp holds
a fragment of the matrix.

* Multiplicand A:

  | .atype | Fragment | Elements (low to high) |
  | --- | --- | --- |
  | `.s4` / `.u4` | A vector expression containing a single `.b32` register, containing eight `.s4` or `.u4` elements from the matrix A. | a0, a1, a2, a3, a4, a5, a6, a7 |

  The layout of the fragments held by different threads is shown in [Figure 59](#mma-8832-a-i4).

  ![_images/mma-8832-A-i4.png](_images/mma-8832-A-i4.png)


  Figure 59 MMA .m8n8k32 fragment layout for matrix A with `.u4`/`.s4` type[](#mma-8832-a-i4 "Permalink to this image")

  The row and column of a matrix fragment can be computed as:

  ```
  groupID = %laneid >> 2

  threadID_in_group = %laneid % 4



  row = groupID



  col = (threadID_in_group * 8) + i for ai where i = {0,..,7}
  ```
* Multiplicand B:

  | .btype | Fragment | Elements (low to high) |
  | --- | --- | --- |
  | `.s4` / `.u4` | A vector expression containing a single `.b32` register, containing eight `.s4` or `.u4` elements from the matrix B. | b0, b1, b2, b3, b4, b5, b6, b7 |

  The layout of the fragments held by different threads is shown in [Figure 60](#mma-8832-b-i4).

  ![_images/mma-8832-B-i4.png](_images/mma-8832-B-i4.png)


  Figure 60 MMA .m8n8k32 fragment layout for matrix B with `.u4`/`.s4` type[](#mma-8832-b-i4 "Permalink to this image")

  The row and column of a matrix fragment can be computed as:

  ```
  groupID = %laneid >> 2

  threadID_in_group = %laneid % 4



  row = (threadID_in_group * 8) + i for bi where i = {0,..,7}



  col = groupID
  ```
* Accumulators (C or D):

  | .ctype / .dtype | Fragment | Elements (low to high) |
  | --- | --- | --- |
  | `.s32` | A vector expression of two `.s32` registers. | c0, c1 |

  The layout of the fragments held by different threads is shown in [Figure 61](#mma-8832-c-i4):

  ![_images/mma-8832-C-i4.png](_images/mma-8832-C-i4.png)


  Figure 61 MMA .m8n8k32 fragment layout for accumulator matrix C/D with `.s32` type[](#mma-8832-c-i4 "Permalink to this image")

  The row and column of a matrix fragment can be computed as:

  ```
  groupID = %laneid >> 2

  threadID_in_group = %laneid % 4



  row = groupID

  col = (threadID_in_group * 2) + i for ci where i = {0, 1}
  ```