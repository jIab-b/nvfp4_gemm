##### 9.7.14.5.8. Matrix Fragments for mma.m16n8k16 with floating point type 

A warp executing `mma.m16n8k16` floating point types will compute an MMA operation of shape
`.m16n8k16`.

Elements of the matrix are distributed across the threads in a warp so each thread of the warp holds
a fragment of the matrix.

* Multiplicand A:

  + `.f16` and `.bf16` :

    > | .atype | Fragment | Elements (low to high) |
    > | --- | --- | --- |
    > | `.f16` / `.bf16` | A vector expression containing four `.f16x2` registers, with each register containing two `.f16` / `.bf16` elements from the matrix A. | a0, a1, a2, a3, a4, a5, a6, a7 |
    >
    > The layout of the fragments held by different threads is shown in [Figure 79](#mma-16816-a-f16).
    >
    > ![_images/mma-16816-A-f16.png](_images/mma-16816-A-f16.png)
    >
    >
    > Figure 79 MMA .m16n8k16 fragment layout for matrix A with `.f16` / `.bf16` type.[](#mma-16816-a-f16 "Permalink to this image")
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
    > row = groupID for ai where 0 <= i < 2 || 4 <= i < 6
    >
    >  groupID + 8 Otherwise
    >
    >
    >
    > col = (threadID_in_group * 2) + (i & 0x1) for ai where i < 4
    >
    > (threadID_in_group * 2) + (i & 0x1) + 8 for ai where i >= 4
    > ```
  + `.f64` :

    > | .atype | Fragment | Elements (low to high) |
    > | --- | --- | --- |
    > | `.f64` | A vector expression containing eight `.f64` registers, with each register containing one `.f64` element from the matrix A. | a0, a1, a2, a3, a4, a5, a6, a7 |
    >
    > The layout of the fragments held by different threads is shown in [Figure 80](#mma-16816-a-f64).
    >
    > ![_images/mma-16816-A-f64.png](_images/mma-16816-A-f64.png)
    >
    >
    > Figure 80 MMA .m16n8k16 fragment layout for matrix A with `.f64` type.[](#mma-16816-a-f64 "Permalink to this image")
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
    > row = groupID for ai where i % 2 = 0
    >
    >  groupID + 8 Otherwise
    >
    >
    >
    > col = (i * 2) + threadID_in_group for ai where i % 2 = 0
    >
    >  (i * 2) - 2 + (threadID_in_group Otherwise
    > ```
* Multiplicand B:

  + `.f16` and `.bf16` :

    > | .btype | Fragment | Elements (low to high) |
    > | --- | --- | --- |
    > | `.f16` / `.bf16` | A vector expression containing two `.f16x2` registers, with each register containing two `.f16` / `.bf16` elements from the matrix B. | b0, b1, b2, b3 |
    >
    > The layout of the fragments held by different threads is shown in [Figure 81](#mma-16816-b-f16).
    >
    > ![_images/mma-16816-B-f16.png](_images/mma-16816-B-f16.png)
    >
    >
    > Figure 81 MMA .m16n8k16 fragment layout for matrix B with `.f16` / `.bf16` type.[](#mma-16816-b-f16 "Permalink to this image")
    >
    > where the row and column of a matrix fragment can be computed as:
    >
    > ```
    > groupID = %laneid >> 2
    >
    > threadID_in_group = %laneid % 4
    >
    >
    >
    > row = (threadID_in_group * 2) + (i & 0x1) for bi where i < 2
    >
    >  (threadID_in_group * 2) + (i & 0x1) + 8 for bi where i >= 2
    >
    >
    >
    > col = groupID
    > ```
  + `.f64` :

    > | .atype | Fragment | Elements (low to high) |
    > | --- | --- | --- |
    > | `.f64` | A vector expression containing four `.f64` registers, with each register containing one `.f64` element from the matrix B. | b0, b1, b2, b3 |
    >
    > The layout of the fragments held by different threads is shown in [Figure 82](#mma-16816-b-f64).
    >
    > ![_images/sparse-mma-16816-tf32-B.png](_images/sparse-mma-16816-tf32-B.png)
    >
    >
    > Figure 82 MMA .m16n8k16 fragment layout for matrix B with `.f64` type.[](#mma-16816-b-f64 "Permalink to this image")
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
    > row = threadID_in_group + (i * 4) for bi where i < 4
    >
    >
    >
    > col = groupID
    > ```
* Accumulators (C or D):

  | .ctype / .dtype | Fragment | Elements (low to high) |
  | --- | --- | --- |
  | `.f64` | A vector expression containing four `.f64` registers containing `.f64` elements from the matrix C (or D). | c0, c1, c2, c3 |
  | `.f32` | A vector expression containing four `.f32` registers containing four `.f32` elements from the matrix C (or D). |
  | `.f16` | A vector expression containing two `.f16x2` registers, with each register containing two `.f16` elements from the matrix C (or D). |

  The layout of the fragments held by different threads is shown in [Figure 83](#mma-16816-c).

  ![_images/mma-16816-C-f16.png](_images/mma-16816-C-f16.png)


  Figure 83 MMA .m16n8k16 fragment layout for accumulator matrix matrix C/D.[](#mma-16816-c "Permalink to this image")

  The row and column of a matrix fragment can be computed as:

  ```
  groupID = %laneid >> 2

  threadID_in_group = %laneid % 4



  row = groupID for ci where i < 2

   groupID + 8 for ci where i >= 2



  col = (threadID_in_group * 2) + (i & 0x1) for ci where i = {0,..,3}
  ```