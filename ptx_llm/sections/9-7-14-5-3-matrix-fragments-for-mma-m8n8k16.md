##### 9.7.14.5.3. Matrix Fragments for mma.m8n8k16 

A warp executing `mma.m8n8k16` will compute an MMA operation of shape `.m8n8k16`.

Elements of the matrix are distributed across the threads in a warp so each thread of the warp holds
a fragment of the matrix.

* Multiplicand A:

  | .atype | Fragment | Elements (low to high) |
  | --- | --- | --- |
  | `.s8` / `.u8` | A vector expression containing a single `.b32` register, containing four `.s8` or `.u8` elements from the matrix A. | a0, a1, a2, a3 |

  The layout of the fragments held by different threads is shown in [Figure 56](#mma-8816-a-i8).

  ![_images/mma-8816-A-i8.png](_images/mma-8816-A-i8.png)


  Figure 56 MMA .m8n8k16 fragment layout for matrix A with `.u8`/`.s8` type[](#mma-8816-a-i8 "Permalink to this image")

  The row and column of a matrix fragment can be computed as:

  ```
  groupID = %laneid >> 2

  threadID_in_group = %laneid % 4



  row = groupID



  col = (threadID_in_group * 4) + i for ai where i = {0,..,3}
  ```
* Multiplicand B:

  | .btype | Fragment | Elements (low to high) |
  | --- | --- | --- |
  | `.s8` / `.u8` | A vector expression containing a single `.b32` register, containing four `.s8` or `.u8` elements from the matrix B. | b0, b1, b2, b3 |

  The layout of the fragments held by different threads is shown in [Figure 57](#mma-8816-b-i8).

  ![_images/mma-8816-B-i8.png](_images/mma-8816-B-i8.png)


  Figure 57 MMA .m8n8k16 fragment layout for matrix B with `.u8`/`.s8` type[](#mma-8816-b-i8 "Permalink to this image")

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
  | `.s32` | A vector expression containing of two `.s32` registers. | c0, c1 |

  The layout of the fragments held by different threads is shown in [Figure 58](#mma-8816-c-i8).

  ![_images/mma-8816-C-i8.png](_images/mma-8816-C-i8.png)


  Figure 58 MMA .m8n8k16 fragment layout for accumulator matrix C/D with `.s32` type[](#mma-8816-c-i8 "Permalink to this image")

  The row and column of a matrix fragment can be computed as:

  ```
  groupID = %laneid >> 2

  threadID_in_group = %laneid % 4



  row = groupID



  col = (threadID_in_group * 2) + i for ci where i = {0, 1}
  ```