###### 9.7.15.6.2.2. Matrix Fragments for sparse wgmma.mma_async.m64nNk16 

A warpgroup executing sparse `wgmma.mma_async.m64nNk16` will compute an MMA operation of shape
`.m64nNk16` where N is a valid n dimension as listed in
[Matrix Shape](#asynchronous-warpgroup-level-matrix-shape).

Elements of the matrix are distributed across the threads in a warpgroup so each thread of the
warpgroup holds a fragment of the matrix.

* Multiplicand A, from shared memory is documented in
  [Shared Memory Matrix Layout](#asynchronous-warpgroup-level-matrix-shared-memory-layout).
* Multiplicand A, from registers:

  > | .atype | Fragments | Elements |
  > | --- | --- | --- |
  > | `.tf32` | A vector expression containing four `.b32`  registers, containing four non-zero `.tf32`  elements out of eight consecutive elements  from matrix A. | Non-zero elements:  a0, a1, a2, a3    Mapping of the non-zero  elements is as described in  [Sparse matrix storage](#asynchronous-warpgroup-level-sparse-matrix-storage) |
  >
  > The layout of the fragments held by different threads is shown in [Figure 177](#sparse-wgmma-64n16-tf32-a).
  >
  > ![_images/sparse-wgmma-64N16-tf32-A.png](_images/sparse-wgmma-64N16-tf32-A.png)
  >
  >
  > Figure 177 Sparse WGMMA .m64nNk16 fragment layout for matrix A with `.tf32` type.[](#sparse-wgmma-64n16-tf32-a "Permalink to this image")
* Accumulator D:

  Matrix fragments for accumulator D are the same as in case of
  [Matrix Fragments for wgmma.mma\_async.m64nNk8](#asynchronous-warpgroup-level-matrix-register-fragment-wgmma-64n8)
  for the same `.dtype` format.
* Multiplicand B:

  Shared memory layout for Matrix B is documented in
  [Shared Memory Matrix Layout](#asynchronous-warpgroup-level-matrix-shared-memory-layout).
* Metadata operand is a `.b32` register containing eight 4-bit vectors each storing the index of a
  non-zero element of a 2-wide chunk of matrix A.

  [Figure 178](#sparse-wgmma-metadata-64n16-tf32) shows the mapping of the metadata bits to the elements
  of matrix A for a warp. In this figure, variable `i` represents the value of the sparsity
  selector operand.

  > ![_images/sparse-mma-metadata-16816-tf32.png](_images/sparse-mma-metadata-16816-tf32.png)
  >
  >
  > Figure 178 Sparse WGMMA .m64nNk16 metadata layout for `.tf32` type.[](#sparse-wgmma-metadata-64n16-tf32 "Permalink to this image")