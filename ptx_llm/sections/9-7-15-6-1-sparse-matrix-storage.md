##### 9.7.15.6.1. Sparse matrix storage 

Granularity of sparse matrix A is defined as the ratio of the number of non-zero elements in a
sub-chunk of the matrix row to the total number of elements in that sub-chunk where the size of the
sub-chunk is shape-specific. For example, in a `64x32` matrix A used in floating point
`wgmma.mma_async` operations, sparsity is expected to be at 2:4 granularity, i.e. each 4-element
vector (i.e. a sub-chunk of 4 consecutive elements) of a matrix row contains 2 zeros. Index of each
non-zero element in a sub-chunk is stored in the metadata operand. Values `0b0000`, `0b0101`,
`0b1010`, `0b1111` are invalid values for metadata and will result in undefined behavior. In a
group of four consecutive threads, one or more threads store the metadata for the whole group
depending upon the matrix shape. These threads are specified using an additional sparsity selector operand.

Matrix A and its corresponding input operand to the sparse wgmma is similar to the diagram shown in
[Figure 111](#sparse-mma-storage-example), with an appropriate matrix size.

Granularities for different matrix shapes and data types are described below.

Sparse `wgmma.mma_async.sp` with half-precision and `.bf16` type

For `.f16` and `.bf16` types, for all supported `64xNx32` shapes, matrix A is structured
sparse at a granularity of 2:4. In other words, each chunk of four adjacent elements in a row of
matrix A have two zeroes and two non-zero elements. Only the two non-zero elements are stored in
matrix A and their positions in the four-wide chunk in Matrix A are indicated by two 2-bits indices
in the metadata operand.

![_images/f16-metadata-example.png](_images/f16-metadata-example.png)


Figure 171 Sparse WGMMA metadata example for `.f16`/`.bf16` type.[](#f16-metadata-example-wgmma "Permalink to this image")

The sparsity selector indicates a thread-pair within a group of four consecutive threads which
contributes the sparsity metadata. Hence, the sparsity selector must be either 0 (threads T0, T1) or
1 (threads T2, T3); any other value results in an undefined behavior.

Sparse `wgmma.mma_async.sp` with `.tf32` type

For `.tf32` type, for all supported `64xNx16` shapes, matrix A is structured sparse at a
granularity of 1:2. In other words, each chunk of two adjacent elements in a row of matrix A have
one zero and one non-zero element. Only the non-zero element is stored in operand for matrix A and
the 4-bit index in the metadata indicates the position of the non-zero element in the two-wide
chunk. 0b1110 and 0b0100 are the only meaningful values of the index, the remaining values result in
an undefined behavior.

![_images/tf32-metadata-example.png](_images/tf32-metadata-example.png)


Figure 172 Sparse WGMMA metadata example for `.tf32` type.[](#tf32-metadata-example-wgmma "Permalink to this image")

The sparsity selector indicates a thread-pair within a group of four consecutive threads which
contributes the sparsity metadata. Hence, the sparsity selector must be either 0 (threads T0, T1) or
1 (threads T2, T3); any other value results in an undefined behavior.

Sparse `wgmma.mma_async.sp` with `.e4m3` and `.e5m2` floating point type

For `.e4m3` and `.e5m2` types, for all supported `64xNx64` shapes, matrix A is structured
sparse at a granularity of 2:4. In other words, each chunk of four adjacent elements in a row of
matrix A have two zeroes and two non-zero elements. Only the two non-zero elements are stored in
matrix A and their positions in the four-wide chunk in Matrix A are indicated by two 2-bits indices
in the metadata operand.

![_images/u8s8-metadata-example.png](_images/u8s8-metadata-example.png)


Figure 173 Sparse WGMMA metadata example for `.e4m3`/`.e5m2` type.[](#e4m3-e5m2-metadata-example-wgmma "Permalink to this image")

All threads contribute the sparsity metadata and the sparsity selector must be 0; any other value
results in an undefined behavior.

Sparse `wgmma.mma_async.sp` with integer type

For the integer type, for all supported `64xNx64` shapes, matrix A is structured sparse at a
granularity of 2:4. In other words, each chunk of four adjacent elements in a row of matrix A have
two zeroes and two non-zero elements. Only the two non-zero elements are stored in matrix A and two
2-bit indices in the metadata indicate the position of these two non-zero elements in the four-wide
chunk.

![_images/u8s8-metadata-example.png](_images/u8s8-metadata-example.png)


Figure 174 Sparse WGMMA metadata example for `.u8`/`.s8` type.[](#u8s8-metadata-example-wgmma "Permalink to this image")

All threads contribute the sparsity metadata and the sparsity selector must be 0; any other value
results in an undefined behavior.