###### 9.7.15.6.2.1. Matrix Fragments for sparse wgmma.mma_async.m64nNk32 

A warpgroup executing sparse `wgmma.mma_async.m64nNk32` will compute an MMA operation of shape
`.m64nNk32` where N is a valid n dimension as listed in
[Matrix Shape](#asynchronous-warpgroup-level-matrix-shape).

Elements of the matrix are distributed across the threads in a warpgroup so each thread of the
warpgroup holds a fragment of the matrix.

* Multiplicand A, from shared memory is documented in
  [Shared Memory Matrix Layout](#asynchronous-warpgroup-level-matrix-shared-memory-layout).
* Multiplicand A, from registers:

  > | .atype | Fragments | Elements |
  > | --- | --- | --- |
  > | `.f16` /  `.bf16` | A vector expression containing four `.b32`  registers, with each register containing two  non-zero `.f16` /`.bf16` elements out of 4  consecutive elements from matrix A. | Non-zero elements:  a0, a1, a2, a3, a4, a5, a6, a7  Mapping of the non-zero  elements is as described in  [Sparse matrix storage](#asynchronous-warpgroup-level-sparse-matrix-storage) |
  >
  > The layout of the fragments held by different threads is shown in [Figure 175](#sparse-wgmma-64n32-f16-bf16-a).
  >
  > ![_images/sparse-wgmma-64N32-f16-bf16-A.png](_images/sparse-wgmma-64N32-f16-bf16-A.png)
  >
  >
  > Figure 175 Sparse WGMMA .m64nNk32 fragment layout for matrix A with `.f16`/`.bf16` type.[](#sparse-wgmma-64n32-f16-bf16-a "Permalink to this image")
* Accumulator D:

  Matrix fragments for accumulator D are the same as in case of
  [Matrix Fragments for wgmma.mma\_async.m64nNk32](#asynchronous-warpgroup-level-matrix-register-fragment-wgmma-64n32)
  for the same `.dtype` format.
* Multiplicand B:

  Shared memory layout for Matrix B is documented in
  [Shared Memory Matrix Layout](#asynchronous-warpgroup-level-matrix-shared-memory-layout).
* Metadata operand is a `.b32` register containing 16 2-bit vectors each storing the index of a
  non-zero element of a 4-wide chunk of matrix A.

  [Figure 176](#sparse-wgmma-metadata-64n32-f16bf16) shows the mapping of the metadata bits to the elements
  of matrix A for a warp. In this figure, variable `i` represents the value of the sparsity
  selector operand.

  > ![_images/sparse-mma-metadata-16832-f16bf16.png](_images/sparse-mma-metadata-16832-f16bf16.png)
  >
  >
  > Figure 176 Sparse WGMMA .m64nNk32 metadata layout for `.f16`/`.bf16` type.[](#sparse-wgmma-metadata-64n32-f16bf16 "Permalink to this image")