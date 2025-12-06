###### 9.7.15.6.2.3. Matrix Fragments for sparse wgmma.mma_async.m64nNk64 

A warpgroup executing sparse `wgmma.mma_async.m64nNk64` will compute an MMA operation of shape
`.m64nNk64` where N is a valid n dimension as listed in
[Matrix Shape](#asynchronous-warpgroup-level-matrix-shape).

Elements of the matrix are distributed across the threads in a warpgroup so each thread of the
warpgroup holds a fragment of the matrix.

* Multiplicand A, from shared memory is documented in
  [Matrix Fragments for sparse wgmma.mma\_async.m64nNk64](#asynchronous-warpgroup-level-matrix-fragment-sparse-wgmma-64n64).
* Multiplicand A, from registers:

  > | .atype | Fragments | Elements |
  > | --- | --- | --- |
  > | `.e4m3` /  `.e5m2` | A vector expression containing four `.b32`  registers, with each register containing four  non-zero `.e4m3` /`.e5m2` elements out of  eight consecutive elements from matrix A. | Non-zero elements:  a0, a1, a2, … , a15    Mapping of the non-zero  elements is as described in  [Sparse matrix storage](#asynchronous-warpgroup-level-sparse-matrix-storage) |
  > | `.s8` /  `.u8` | A vector expression containing four `.b32`  registers, with each register containing four  non-zero `.s8` /`.u8` elements out of  eight consecutive elements from matrix A. |
  >
  > The layout of the fragments held by different threads is shown in [Figure 179](#sparse-wgmma-64n64-e4m3-e5m2-s8-u8-a).
  >
  > ![_images/sparse-wgmma-64N64-e4m3-e5m2-s8-u8-A.png](_images/sparse-wgmma-64N64-e4m3-e5m2-s8-u8-A.png)
  >
  >
  > Figure 179 Sparse WGMMA .m64nNk64 fragment layout for matrix A with `.e4m3`/ `.e5m2`/ `.s8`/ `.u8` type.[](#sparse-wgmma-64n64-e4m3-e5m2-s8-u8-a "Permalink to this image")
* Accumulator D:

  Matrix fragments for accumulator D are the same as in case of
  [Matrix Fragments for wgmma.mma\_async.m64nNk32](#asynchronous-warpgroup-level-matrix-register-fragment-wgmma-64n32)
  for the same `.dtype` format.
* Multiplicand B:

  Shared memory layout for Matrix B is documented in
  [Matrix Fragments for sparse wgmma.mma\_async.m64nNk64](#asynchronous-warpgroup-level-matrix-fragment-sparse-wgmma-64n64).
* Metadata operand is a `.b32` register containing 16 4-bit vectors each storing the indices of
  two non-zero elements of a 4-wide chunk of matrix A.

  [Figure 180](#sparse-wgmma-metadata-64n64-e4m3-e5m2-s8-u8-first32col) shows the mapping of the metadata
  bits to the elements of columns 0–31 of matrix A.

  > ![_images/sparse-mma-metadata-16864-u8s8-first32col.png](_images/sparse-mma-metadata-16864-u8s8-first32col.png)
  >
  >
  > Figure 180 Sparse WGMMA .m64nNk64 metadata layout for `.e4m3`/ `.e5m2`/ `.s8`/ `.u8` type for columns 0–31[](#sparse-wgmma-metadata-64n64-e4m3-e5m2-s8-u8-first32col "Permalink to this image")

  [Figure 181](#sparse-wgmma-metadata-64n64-e4m3-e5m2-s8-u8-last32col) shows the mapping of the metadata
  bits to the elements of columns 32–63 of matrix A.

  > ![_images/sparse-mma-metadata-16864-u8s8-last32col.png](_images/sparse-mma-metadata-16864-u8s8-last32col.png)
  >
  >
  > Figure 181 Sparse WGMMA .m64nNk64 metadata layout for `.e4m3`/ `.e5m2`/ `.s8`/ `.u8` type for columns 32–63[](#sparse-wgmma-metadata-64n64-e4m3-e5m2-s8-u8-last32col "Permalink to this image")