##### 9.7.14.5.10. Matrix Fragments for mma.m16n8k32 

A warp executing `mma.m16n8k32` will compute an MMA operation of shape `.m16n8k32`.

Elements of the matrix are distributed across the threads in a warp so each thread of the warp holds
a fragment of the matrix.

* Multiplicand A:

  + `.s4` or `.u4` :

    > | .atype | Fragment | Elements (low to high) |
    > | --- | --- | --- |
    > | `.s4` / `.u4` | A vector expression containing two `.b32` registers, with each register containing eight `.u4` / `.s4` elements from the matrix A. | a0, a1, …, a14, a15 |
    >
    > The layout of the fragments held by different threads is shown in [Figure 87](#mma-16832-a-i4).
    >
    > ![_images/mma-16832-A-i4.png](_images/mma-16832-A-i4.png)
    >
    >
    > Figure 87 MMA .m16n8k32 fragment layout for matrix A with `.u4` / `.s4` type.[](#mma-16832-a-i4 "Permalink to this image")
    >
    > The row and column of a matrix fragment can be computed as:
    >
    > ```
    > groupID = %laneid >> 2
    >
    > threadID_in_group = %laneid % 4
    >
    >
    >
    > row = groupID for ai where i < 8
    >
    >  groupID + 8 for ai where i >= 8
    >
    >
    >
    > col = (threadID_in_group * 8) + (i & 0x7) for ai where i = {0,..,15}
    > ```
  + `.s8` or `.u8` or `.e4m3` or `.e5m2` or `.e3m2` or `.e2m3` or `.e2m1`:

    > | .atype | Fragment | Elements (low to high) |
    > | --- | --- | --- |
    > | `.s8` / `.u8` | A vector expression containing four `.b32` registers, with each register containing four `.s8` / `.u8` elements from the matrix A. | a0, a1, .., a14, a15 |
    > | `.e4m3` / `.e5m2` / `.e3m2` / `.e2m3` / `.e2m1` | A vector expression containing four `.b32` registers, with each register containing four `.e4m3` / `.e5m2` / `.e3m2` / `.e2m3` / `.e2m1` elements from the matrix A. | a0, a1, …, a14, a15 |
    >
    > The layout of the fragments held by different threads is shown in [Figure 88](#mma-16832-a-i8).
    >
    > ![_images/mma-16832-A-i8.png](_images/mma-16832-A-i8.png)
    >
    >
    > Figure 88 MMA .m16n8k32 fragment layout for matrix A with `.u8` / `.s8` / `.e4m3` / `.e5m2` / `.e3m2` / `.e2m3` / `.e2m1` type.[](#mma-16832-a-i8 "Permalink to this image")
    >
    > The row and column of a matrix fragment can be computed as:
    >
    > ```
    > groupID = %laneid >> 2
    >
    > threadID_in_group = %laneid % 4
    >
    >
    >
    > row = groupID for ai where 0 <= i < 4 || 8 <= i < 12
    >
    >  groupID + 8 otherwise
    >
    >
    >
    > col = (threadID_in_group * 4) + (i & 0x3) for ai where i < 8
    >
    >  (threadID_in_group * 4) + (i & 0x3) + 16 for ai where i >= 8
    > ```
* Multiplicand B:

  + `.s4` or `.u4` :

    > | .btype | Fragment | Elements (low to high) |
    > | --- | --- | --- |
    > | `.s4` / `.u4` | A vector expression containing a single `.b32` register, containing eight `.s4` / `.u4` elements from the matrix B. | b0, b1, b2, b3, b4, b5, b6, b7 |
    >
    > The layout of the fragments held by different threads is shown in [Figure 89](#mma-16832-b-i4).
    >
    > ![_images/mma-16832-B-i4.png](_images/mma-16832-B-i4.png)
    >
    >
    > Figure 89 MMA .m16n8k32 fragment layout for matrix B with `.u4` / `.s4` type.[](#mma-16832-b-i4 "Permalink to this image")
    >
    > The row and column of a matrix fragment can be computed as:

  ```
  groupID = %laneid >> 2

  threadID_in_group = %laneid % 4



  row = (threadID_in_group * 8) + (i & 0x7) for bi where i = {0,..,7}

  col = groupID
  ```

  + `.s8` or `.u8` or `.e4m3` or `.e5m2` or `.e3m2` or `.e2m3` or `.e2m1`:

    > | .btype | Fragment | Elements (low to high) |
    > | --- | --- | --- |
    > | `.s8` / `.u8` | A vector expression containing two `.b32` registers, with each register containing four `.s8` / `.u8` elements from the matrix B. | b0, b1, b2, b3, b4, b5, b6, b7 |
    > | `.e4m3` / `.e5m2` / `.e3m2` / `.e2m3` / `.e2m1` | A vector expression containing two `.b32` registers, with each register containing four `.e4m3` / `.e5m2` / `.e3m2` / `.e2m3` / `.e2m1` elements from the matrix B. | b0, b1, b2, b3, b4, b5, b6, b7 |
    >
    > The layout of the fragments held by different threads is shown in [Figure 90](#mma-16832-b-i8-1) and
    > [Figure 91](#mma-16832-b-i8-2).
    >
    > ![_images/mma-16832-B-i8_1.png](_images/mma-16832-B-i8_1.png)
    >
    >
    > Figure 90 MMA .m16n8k32 fragment layout for rows 0–15 of matrix B with `.u8` / `.s8` / `.e4m3` / `.e5m2` / `.e3m2` / `.e2m3` / `.e2m1` type.[](#mma-16832-b-i8-1 "Permalink to this image")
    >
    >
    > ![_images/mma-16832-B-i8_2.png](_images/mma-16832-B-i8_2.png)
    >
    >
    > Figure 91 MMA .m16n8k32 fragment layout for rows 16–31 of matrix B with `.u8` / `.s8` / `.e4m3` / `.e5m2` / `.e3m2` / `.e2m3` / `.e2m1` type.[](#mma-16832-b-i8-2 "Permalink to this image")
    >
    > The row and column of a matrix fragment can be computed as:
    >
    > ```
    > groupID = %laneid >> 2
    >
    > threadID_in_group = %laneid % 4
    >
    >
    >
    > row = (threadID_in_group * 4) + (i & 0x3) for bi where i < 4
    >
    >  (threadID_in_group * 4) + (i & 0x3) + 16 for bi where i >= 4
    >
    >
    >
    > col = groupID
    > ```
* Accumulators (C or D):

  | .ctype / .dtype | Fragment | Elements (low to high) |
  | --- | --- | --- |
  | `.s32` | A vector expression containing four `.s32` registers, containing four `.s32` elements from the matrix C (or D). | c0, c1, c2, c3 |
  | `.f32` | A vector expression containing four `.f32` registers, containing four `.f32` elements from the matrix C (or D). | c0, c1, c2, c3 |
  | `.f16` | A vector expression containing two `.f16x2` registers, with each register containing two `.f16` elements from the matrix C (or D). | c0, c1, c2, c3 |

  The layout of the fragments held by different threads is shown in [Figure 92](#mma-16832-c).

  ![_images/mma-16832-C.png](_images/mma-16832-C.png)


  Figure 92 MMA .m16n8k32 fragment layout for accumulator matrix C/D with `.s32` / `.f32` / `.f16` type.[](#mma-16832-c "Permalink to this image")

  The row and column of a matrix fragment can be computed as:

  ```
  groupID = %laneid >> 2

  threadID_in_group = %laneid % 4



  row = groupID for ci where i < 2

   groupID + 8 for ci where i >= 2



  col = (threadID_in_group * 2) + (i & 0x1) for ci where i = {0,..,3}
  ```