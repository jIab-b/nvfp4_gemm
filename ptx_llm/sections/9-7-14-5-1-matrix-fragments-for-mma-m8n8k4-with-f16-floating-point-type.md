##### 9.7.14.5.1. Matrix Fragments for mma.m8n8k4 with .f16 floating point type 

A warp executing `mma.m8n8k4` with `.f16` floating point type will compute 4 MMA operations of shape
`.m8n8k4`.

Elements of 4 matrices need to be distributed across the threads in a warp. The following table
shows distribution of matrices for MMA operations.

| MMA Computation | Threads participating in MMA computation |
| --- | --- |
| MMA computation 1 | Threads with `%laneid` 0-3 (low group) and 16-19 (high group) |
| MMA computation 2 | Threads with `%laneid` 4-7 (low group) and 20-23 (high group) |
| MMA computation 3 | Threads with `%laneid` 8-11 (low group) and 24-27 (high group) |
| MMA computation 4 | Threads with `%laneid` 12-15 (low group) and 28-31 (high group) |

For each of the individual MMA computation shown above, each of the required thread holds a fragment
of the matrix for performing mma operation as follows:

* Multiplicand A:

  | .atype | Fragment | Elements (low to high) |
  | --- | --- | --- |
  | `.f16` | A vector expression containing two `.f16x2` registers, with each register containing two `.f16` elements from the matrix A. | a0, a1, a2, a3 |

  The layout of the fragments held by different threads is shown below:

  + Fragment layout for Row Major matrix A is shown in [Figure 46](#mma-884-a-row-f16).

    ![_images/mma-884-A-row-f16.png](_images/mma-884-A-row-f16.png)


    Figure 46 MMA .m8n8k4 fragment layout for row-major matrix A with `.f16` type[](#mma-884-a-row-f16 "Permalink to this image")

    The row and column of a matrix fragment can be computed as:

    ```
    row = %laneid % 4 if %laneid < 16

     (%laneid % 4) + 4 otherwise



    col = i for ai where i = {0,..,3}
    ```
  + Fragment layout for Column Major matrix A is shown in [Figure 47](#mma-884-a-col-f16).

    The layout of the fragments held by different threads is shown below:

    ![_images/mma-884-A-col-f16.png](_images/mma-884-A-col-f16.png)


    Figure 47 MMA .m8n8k4 fragment layout for column-major matrix A with `.f16` type[](#mma-884-a-col-f16 "Permalink to this image")

    The row and column of a matrix fragment can be computed as:

    ```
    row = i % 4 for ai where i = {0,..,3} if %laneid < 16

     (i % 4) + 4 for ai where i = {0,..,3} otherwise



    col = %laneid % 4
    ```
* Multiplicand B:

  | .btype | Fragment | Elements (low to high) |
  | --- | --- | --- |
  | `.f16` | A vector expression containing two `.f16x2` registers, with each register containing two `.f16` elements from the matrix B. | b0, b1, b2, b3 |

  The layout of the fragments held by different threads is shown below:

  + Fragment layout for Row Major matrix B is shown in [Figure 48](#mma-884-b-row-f16).

    ![_images/mma-884-B-row-f16.png](_images/mma-884-B-row-f16.png)


    Figure 48 MMA .m8n8k4 fragment layout for row-major matrix B with `.f16` type[](#mma-884-b-row-f16 "Permalink to this image")

    The row and column of a matrix fragment can be computed as:

    ```
    row = %laneid % 4



    col = i for bi where i = {0,..,3} if %laneid < 16

     i+4 for bi where i = {0,..,3} otherwise
    ```
  + Fragment layout for Column Major matrix B is shown in [Figure 49](#mma-884-b-col-f16).

    ![_images/mma-884-B-col-f16.png](_images/mma-884-B-col-f16.png)


    Figure 49 MMA .m8n8k4 fragment layout for column-major matrix B with `.f16` type[](#mma-884-b-col-f16 "Permalink to this image")

    The row and column of a matrix fragment can be computed as:

    ```
    row = i for bi where i = {0,..,3}



    col = %laneid % 4 if %laneid < 16

     (%laneid % 4) + 4 otherwise
    ```
* Accumulators C (or D):

  | .ctype / .dtype | Fragment | Elements (low to high) |
  | --- | --- | --- |
  | `.f16` | A vector expression containing four `.f16x2` registers, with each register containing two `.f16` elements from the matrix C (or D). | c0, c1, c2, c3, c4, c5, c6, c7 |
  | `.f32` | A vector expression of eight `.f32` registers. |

  The layout of the fragments held by different threads is shown below:

  + Fragment layout for accumulator matrix when `.ctype` is `.f16` is shown in [Figure 50](#mma-884-c-f16).

    ![_images/mma-884-C-f16.png](_images/mma-884-C-f16.png)


    Figure 50 MMA .m8n8k4 fragment layout for matrix C/D with `.ctype` = `.f16`[](#mma-884-c-f16 "Permalink to this image")

    The row and column of a matrix fragment can be computed as:

    ```
    row = %laneid % 4 if %laneid < 16

     (%laneid % 4) + 4 otherwise



    col = i for ci where i = {0,..,7}
    ```
  + Fragment layout for accumulator matrix when `.ctype` is `.f32` is shown in
    [Figure 51](#mma-884-c-f32-1) and [Figure 52](#mma-884-c-f32-2).

    ![_images/mma-884-C-f32-1.png](_images/mma-884-C-f32-1.png)


    Figure 51 MMA .m8n8k4 computation 1 and 2 fragment layout for matrix C/D with `.ctype` = `.f32`[](#mma-884-c-f32-1 "Permalink to this image")


    ![_images/mma-884-C-f32-2.png](_images/mma-884-C-f32-2.png)


    Figure 52 MMA .m8n8k4 computation 3 and 4 fragment layout for matrix C/D with `.ctype` = `.f32`[](#mma-884-c-f32-2 "Permalink to this image")

    The row and column of a matrix fragment can be computed as:

    ```
    row = X if %laneid < 16

     X + 4 otherwise



     where X = (%laneid & 0b1) + (i & 0b10) for ci where i = {0,..,7}



    col = (i & 0b100) + (%laneid & 0b10) + (i & 0b1) for ci where i = {0,..,7}
    ```