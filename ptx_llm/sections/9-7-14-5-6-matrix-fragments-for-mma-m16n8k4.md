##### 9.7.14.5.6. Matrix Fragments for mma.m16n8k4 

A warp executing `mma.m16n8k4` will compute an MMA operation of shape `.m16n8k4`.

Elements of the matrix are distributed across the threads in a warp so each thread of the warp holds
a fragment of the matrix.

* Multiplicand A:

  + `.tf32`:

    | .atype | Fragment | Elements (low to high) |
    | --- | --- | --- |
    | `.tf32` | A vector expression containing two `.b32` registers, containing two `.tf32` elements from the matrix A. | a0, a1 |

    The layout of the fragments held by different threads is shown in [Figure 65](#mma-1684-a-tf32).

    ![_images/mma-1684-A.png](_images/mma-1684-A.png)


    Figure 65 MMA .m16n8k4 fragment layout for matrix A with `.tf32` type.[](#mma-1684-a-tf32 "Permalink to this image")

    The row and column of a matrix fragment can be computed as:

    ```
    groupID = %laneid >> 2

    threadID_in_group = %laneid % 4



    row = groupID for a0

     groupID + 8 for a1



    col = threadID_in_group
    ```
  + `.f64`:

    > | .atype | Fragment | Elements (low to high) |
    > | --- | --- | --- |
    > | `.f64` | A vector expression containing two `.f64` registers, containing two `.f64` elements from the matrix A. | a0, a1 |
    >
    > The layout of the fragments held by different threads is shown in [Figure 66](#mma-1684-a-f64).
    >
    > ![_images/mma-1684-A.png](_images/mma-1684-A.png)
    >
    >
    > Figure 66 MMA .m16n8k4 fragment layout for matrix A with `.f64` type.[](#mma-1684-a-f64 "Permalink to this image")
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
    > row = groupID for a0
    >
    >  groupID + 8 for a1
    >
    >
    >
    > col = threadID_in_group
    > ```
* Multiplicand B:

  + `.tf32`:

    > | .btype | Fragment | Elements (low to high) |
    > | --- | --- | --- |
    > | `.tf32` | A vector expression of a single `.b32` register, containing a single `.tf32` element from the matrix B. | b0 |
    >
    > The layout of the fragments held by different threads is shown in [Figure 67](#mma-1684-b-tf32).
    >
    > ![_images/mma-1684-B.png](_images/mma-1684-B.png)
    >
    >
    > Figure 67 MMA .m16n8k4 fragment layout for matrix B with `.tf32` type.[](#mma-1684-b-tf32 "Permalink to this image")
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
    > row = threadID_in_group
    >
    >
    >
    > col = groupID
    > ```
  + `.f64`:

    > | .btype | Fragment | Elements (low to high) |
    > | --- | --- | --- |
    > | `.f64` | A vector expression of a single `.f64` register, containing a single `.f64` element from the matrix B. | b0 |
    >
    > The layout of the fragments held by different threads is shown in [Figure 68](#mma-1684-b-f64).
    >
    > ![_images/mma-1684-B.png](_images/mma-1684-B.png)
    >
    >
    > Figure 68 MMA .m16n8k4 fragment layout for matrix B with `.f64` type.[](#mma-1684-b-f64 "Permalink to this image")
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
    > row = threadID_in_group
    >
    >
    >
    > col = groupID
    > ```
* Accumulators (C or D):

  + `.tf32`:

    > | .ctype / .dtype | Fragment | Elements (low to high) |
    > | --- | --- | --- |
    > | `.f32` | A vector expression containing four `.f32` registers, containing four `.f32` elements from the matrix C (or D). | c0, c1, c2, c3 |
    >
    > The layout of the fragments held by different threads is shown in [Figure 69](#mma-1684-c-f32).
    >
    > ![_images/mma-1684-C.png](_images/mma-1684-C.png)
    >
    >
    > Figure 69 MMA .m16n8k4 fragment layout for accumulator matrix C/D with `.f32` type.[](#mma-1684-c-f32 "Permalink to this image")
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
  + `.f64`:

    > | .ctype / .dtype | Fragment | Elements (low to high) |
    > | --- | --- | --- |
    > | `.f64` | A vector expression containing four `.f64` registers, containing four `.f64` elements from the matrix C (or D). | c0, c1, c2, c3 |
    >
    > The layout of the fragments held by different threads is shown in [Figure 70](#mma-1684-c-f64).
    >
    > ![_images/mma-1684-C.png](_images/mma-1684-C.png)
    >
    >
    > Figure 70 MMA .m16n8k4 fragment layout for accumulator matrix C/D with `.f64` type.[](#mma-1684-c-f64 "Permalink to this image")
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