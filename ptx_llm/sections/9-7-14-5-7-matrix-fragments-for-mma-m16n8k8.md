##### 9.7.14.5.7. Matrix Fragments for mma.m16n8k8 

A warp executing `mma.m16n8k8` will compute an MMA operation of shape `.m16n8k8`.

Elements of the matrix are distributed across the threads in a warp so each thread of the warp holds
a fragment of the matrix.

* Multiplicand A:

  + `.f16` and `.bf16` :

    > | .atype | Fragment | Elements (low to high) |
    > | --- | --- | --- |
    > | `.f16` / `.bf16` | A vector expression containing two `.f16x2` registers, with each register containing two `.f16` / `.bf16` elements from the matrix A. | a0, a1, a2, a3 |
    >
    > The layout of the fragments held by different threads is shown in [Figure 71](#mma-1688-a-f16).
    >
    > ![_images/mma-1688-A-f16.png](_images/mma-1688-A-f16.png)
    >
    >
    > Figure 71 MMA .m16n8k8 fragment layout for matrix A with `.f16` / `.bf16` type.[](#mma-1688-a-f16 "Permalink to this image")
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
    > row = groupID for a0 and a1
    >
    >  groupID + 8 for a2 and a3
    >
    >
    >
    > col = threadID_in_group * 2 + (i & 0x1) for ai where i = {0,..,3}
    > ```
  + `.tf32` :

    > | .atype | Fragment | Elements (low to high) |
    > | --- | --- | --- |
    > | `.tf32` | A vector expression containing four `.b32` registers, containing four `.tf32` elements from the matrix A. | a0, a1, a2, a3 |
    >
    > The layout of the fragments held by different threads is shown in [Figure 72](#mma-1688-a-tf32).
    >
    > ![_images/mma-1688-A-tf32.png](_images/mma-1688-A-tf32.png)
    >
    >
    > Figure 72 MMA .m16n8k8 fragment layout for matrix A with `.tf32` type.[](#mma-1688-a-tf32 "Permalink to this image")
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
    > row = groupID for a0 and a2
    >
    >  groupID + 8 for a1 and a3
    >
    >
    >
    > col = threadID_in_group for a0 and a1
    >
    >  threadID_in_group + 4 for a2 and a3
    > ```
  + `.f64` :

    | .atype | Fragment | Elements (low to high) |
    | --- | --- | --- |
    | `.f64` | A vector expression containing four `.f64` registers, containing four `.f64` elements from the matrix A. | a0, a1, a2, a3 |

    The layout of the fragments held by different threads is shown in [Figure 73](#mma-1688-a-f64).

    ![_images/mma-1688-A-tf32.png](_images/mma-1688-A-tf32.png)


    Figure 73 MMA .m16n8k8 fragment layout for matrix A with `.f64` type.[](#mma-1688-a-f64 "Permalink to this image")

    The row and column of a matrix fragment can be computed as:

    ```
    groupID = %laneid >> 2

    threadID_in_group = %laneid % 4



    row = groupID for a0 and a2

     groupID + 8 for a1 and a3



    col = threadID_in_group for a0 and a1

     threadID_in_group + 4 for a2 and a3
    ```
* Multiplicand B:

  + `.f16` and `.bf16` :

    > | .btype | Fragment | Elements (low to high) |
    > | --- | --- | --- |
    > | `.f16` / `.bf16` | A vector expression containing a single `.f16x2` register, containing two `.f16` / `.bf16` elements from the matrix B. | b0, b1 |
    >
    > The layout of the fragments held by different threads is shown in [Figure 74](#mma-1688-b-f16).
    >
    > ![_images/mma-1688-B-f16.png](_images/mma-1688-B-f16.png)
    >
    >
    > Figure 74 MMA .m16n8k8 fragment layout for matrix B with `.f16` / `.bf16` type.[](#mma-1688-b-f16 "Permalink to this image")
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
    > row = (threadID_in_group * 2) + i for bi where i = {0, 1}
    >
    >
    >
    > col = groupID
    > ```
  + `.tf32` :

    > | .btype | Fragment | Elements (low to high) |
    > | --- | --- | --- |
    > | `.tf32` | A vector expression containing two `.b32` registers, containing two `.tf32` elements from the matrix B. | b0, b1 |
    >
    > The layout of the fragments held by different threads is shown in [Figure 75](#mma-1688-b-tf32).
    >
    > ![_images/mma-1688-B-tf32.png](_images/mma-1688-B-tf32.png)
    >
    >
    > Figure 75 MMA .m16n8k8 fragment layout for matrix B with `.tf32` type.[](#mma-1688-b-tf32 "Permalink to this image")
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
    > row = threadID_in_group for b0
    >
    >  threadID_in_group + 4 for b1
    >
    >
    >
    > col = groupID
    > ```
  + `.f64` :

    > | .btype | Fragment | Elements (low to high) |
    > | --- | --- | --- |
    > | `.f64` | A vector expression containing two `.f64` registers, containing two `.f64` elements from the matrix B. | b0, b1 |
    >
    > The layout of the fragments held by different threads is shown in [Figure 76](#mma-1688-b-f64).
    >
    > ![_images/mma-1688-B-tf32.png](_images/mma-1688-B-tf32.png)
    >
    >
    > Figure 76 MMA .m16n8k8 fragment layout for matrix B with `.f64` type.[](#mma-1688-b-f64 "Permalink to this image")
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
    > row = threadID_in_group for b0
    >
    >  threadID_in_group + 4 for b1
    >
    >
    >
    > col = groupID
    > ```
* Accumulators (C or D):

  + `.f16`, `.bf16` and `.tf32`:

    > | .ctype / .dtype | Fragment | Elements (low to high) |
    > | --- | --- | --- |
    > | `.f16` | A vector expression containing two `.f16x2` registers, with each register containing two `.f16` elements from the matrix C (or D). | c0, c1, c2, c3 |
    > | `.f32` | A vector expression of four `.f32` registers. |  |
    >
    > The layout of the fragments held by different threads is shown in [Figure 77](#mma-1688-c-f16-f32).
    >
    > ![_images/mma-1688-C.png](_images/mma-1688-C.png)
    >
    >
    > Figure 77 MMA .m16n8k8 fragment layout for accumulator matrix C/D with `.f16x2`/`.f32` type.[](#mma-1688-c-f16-f32 "Permalink to this image")
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
    > row = groupID for c0 and c1
    >
    >  groupID + 8 for c2 and c3
    >
    >
    >
    > col = (threadID_in_group * 2) + (i & 0x1) for ci where i = {0,..,3}
    > ```
  + `.f64` :

    > | .ctype / .dtype | Fragment | Elements (low to high) |
    > | --- | --- | --- |
    > | `.f64` | A vector expression of four `.f64` registers containing four `.f64` elements from the matrix C (or D). | c0, c1, c2, c3 |
    >
    > The layout of the fragments held by different threads is shown in [Figure 78](#mma-1688-c-f64).
    >
    > ![_images/mma-1688-C.png](_images/mma-1688-C.png)
    >
    >
    > Figure 78 MMA .m16n8k8 fragment layout for accumulator matrix C/D with `.f64` type.[](#mma-1688-c-f64 "Permalink to this image")
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
    > row = groupID for c0 and c1
    >
    >  groupID + 8 for c2 and c3
    >
    >
    >
    > col = (threadID_in_group * 2) + (i & 0x1) for ci where i = {0,..,3}
    > ```