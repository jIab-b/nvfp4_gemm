##### 9.7.14.6.1. Sparse matrix storage 

Granularity of sparse matrix A is defined as the ratio of the number of non-zero elements in a
sub-chunk of the matrix row to the total number of elements in that sub-chunk where the size of the
sub-chunk is shape-specific. For example, in a `16x16` matrix A, sparsity is expected to be at 2:4
granularity, i.e. each 4-element vector (i.e. a sub-chunk of 4 consecutive elements) of a matrix row
contains 2 zeros. Index of each non-zero element in a sub-chunk is stored in the metadata
operand. Values `0b0000`, `0b0101`, `0b1010`, `0b1111` are invalid values for metadata and
will result in undefined behavior. In a group of four consecutive threads, one or more threads store
the metadata for the whole group depending upon the matrix shape. These threads are specified using
an additional *sparsity selector* operand.

[Figure 111](#sparse-mma-storage-example) shows an example of a 16x16 matrix A represented in sparse format and sparsity
selector indicating which thread in a group of four consecutive threads stores the metadata.

![_images/sparse-mma-storage-example.png](_images/sparse-mma-storage-example.png)


Figure 111 Sparse MMA storage example[](#sparse-mma-storage-example "Permalink to this image")

Granularities for different matrix shapes and data types are described below.

Sparse `mma.sp{::ordered_metadata}` with half-precision and `.bf16` type

For the `.m16n8k16` and `.m16n8k32` `mma.sp{::ordered_metadata}` operations, matrix A is
structured sparse at a granularity of 2:4. In other words, each chunk of four adjacent elements
in a row of matrix A has two zeros and two non-zero elements. Only the two non-zero elements are
stored in the operand representing matrix A and their positions in the four-wide chunk in matrix
A are indicated by two 2-bit indices in the metadata operand. For `mma.sp::ordered_metadata`,
`0b0100`, `0b1000`, `0b1001`, `0b1100`, `0b1101`, `0b1110` are the meaningful values
of indices; any other values result in an undefined behavior.

![_images/f16-metadata-example.png](_images/f16-metadata-example.png)


Figure 112 Sparse MMA metadata example for `.f16`/`.bf16` type.[](#f16-metadata-example "Permalink to this image")

The sparsity selector indicates the threads which contribute metadata as listed below:

* `m16n8k16`: One thread within a group of four consecutive threads contributes the metadata for
  the entire group. This thread is indicated by a value in {0, 1, 2, 3}.
* `m16n8k32`: A thread-pair within a group of four consecutive threads contributes the sparsity
  metadata. Hence, the sparsity selector must be either 0 (threads T0, T1) or 1 (threads T2, T3);
  any other value results in an undefined behavior.

Sparse `mma.sp{::ordered_metadata}` with `.tf32` type

When matrix A has `.tf32` elements, matrix A is structured sparse at a granularity of 1:2. In
other words, each chunk of two adjacent elements in a row of matrix A has one zero and one non-zero
element. Only the non-zero elements are stored in the operand for matrix A and their positions in a
two-wide chunk in matrix A are indicated by the 4-bit index in the metadata. `0b1110` and
`0b0100` are the only meaningful index values; any other values result in an undefined behavior.

![_images/tf32-metadata-example.png](_images/tf32-metadata-example.png)


Figure 113 Sparse MMA metadata example for `.tf32` type.[](#tf32-metadata-example "Permalink to this image")

The sparsity selector indicates the threads which contribute metadata as listed below:

* `m16n8k8`: One thread within a group of four consecutive threads contributes the metadata for
  the entire group. This thread is indicated by a value in {0, 1, 2, 3}.
* `m16n8k16`: A thread-pair within a group of four consecutive threads contributes the sparsity
  metadata. Hence, the sparsity selector must be either 0 (threads T0, T1) or 1 (threads T2, T3);
  any other value results in an undefined behavior.

Sparse `mma.sp{::ordered_metadata}` with integer type

When matrices A and B have `.u8`/`.s8` elements, matrix A is structured sparse at a granularity
of 2:4. In other words, each chunk of four adjacent elements in a row of matrix A have two zeroes
and two non-zero elements. Only the two non-zero elements are stored in sparse matrix and their
positions in the four-wide chunk are indicated by two 2-bit indices in the metadata. For
`mma.sp::ordered_metadata`, `0b0100`, `0b1000`, `0b1001`, `0b1100`, `0b1101`, `0b1110`
are the meaningful values of indices; any other values result in an undefined behavior.

![_images/u8s8-metadata-example.png](_images/u8s8-metadata-example.png)


Figure 114 Sparse MMA metadata example for `.u8`/`.s8` type.[](#u8s8-metadata-example "Permalink to this image")

when matrices A and B have `.u4`/`.s4` elements, matrix A is pair-wise structured sparse at a
granularity of 4:8. In other words, each chunk of eight adjacent elements in a row of matrix A has
four zeroes and four non-zero values. Further, the zero and non-zero values are clustered in
sub-chunks of two elements each within the eight-wide chunk. i.e., each two-wide sub-chunk within
the eight-wide chunk must be all zeroes or all non-zeros. Only the four non-zero values are stored
in sparse matrix and the positions of the two two-wide sub-chunks with non-zero values in the
eight-wide chunk of a row of matrix A are indicated by two 2-bit indices in the metadata. For
`mma.sp::ordered_metadata`, `0b0100`, `0b1000`, `0b1001`, `0b1100`, `0b1101`, `0b1110`
are the meaningful values of indices; any other values result in an undefined behavior.

![_images/u4s4-metadata-example.png](_images/u4s4-metadata-example.png)


Figure 115 Sparse MMA metadata example for `.u4`/`.s4` type.[](#u4s4-metadata-example "Permalink to this image")

The sparsity selector indicates the threads which contribute metadata as listed below:

* `m16n8k32` with `.u8`/`.s8` type and `m16n8k64` with `.u4`/`.s4` type: A thread-pair
  within a group of four consecutive threads contributes the sparsity metadata. Hence, the sparsity
  selector must be either 0 (threads T0, T1) or 1 (threads T2, T3); any other value results in an
  undefined behavior.
* `m16n8k64` with `.u8`/`.s8` type and `m16n8k128` with `.u4`/`.s4` type: All threads
  within a group of four consecutive threads contribute the sparsity metadata. Hence, the sparsity
  selector in this case must be 0. Any other value of sparsity selector results in an undefined
  behavior.

Sparse `mma.sp{::ordered_metadata}` operating on `.e4m3`/`.e5m2`/`.e3m2`/`.e2m3`/`.e2m1`
type with `.kind::f8f6f4` or `.kind::mxf8f6f4`

When matrices A and B have `.e4m3`/`.e5m2`/`.e3m2`/`.e2m3`/`.e2m1` elements, matrix A is
structured sparse at a granularity of 2:4. In other words, each chunk of four adjacent elements in a
row of matrix A have two zeroes and two non-zero elements. Only the two non-zero elements are stored
in sparse matrix and their positions in the four-wide chunk are indicated by two 2-bit indices in the
metadata. `0b0100`, `0b1000`, `0b1001`, `0b1100`, `0b1101`, `0b1110` are the meaningful
values of indices; any other values result in an undefined behavior.

![_images/fp8-metadata-example.png](_images/fp8-metadata-example.png)


Figure 116 Sparse MMA metadata example for `.e4m3`/`.e5m2`/`.e3m2`/`.e2m3`/`.e2m1` type.[](#fp8-metadata-example "Permalink to this image")

The sparsity selector indicates the threads which contribute metadata as listed below:

* `m16n8k64`: All threads within a group of four consecutive threads contribute the sparsity metadata.
  Hence, the sparsity selector in this case must be 0. Any other value of sparsity selector results in
  an undefined behavior.

Sparse `mma.sp::ordered_metadata` operating on `.e2m1` type with `.kind::mxf4` or `.kind::mxf4nvf4`

When matrices A and B have `.e2m1` elements, matrix A is pair-wise structured sparse at a granularity
of 4:8. In other words, each chunk of eight adjacent elements in a row of matrix A has four zeroes and
four non-zero values. Further, the zero and non-zero values are clustered in sub-chunks of two elements
each within the eight-wide chunk. i.e., each two-wide sub-chunk within the eight-wide chunk must be all
zeroes or all non-zeros. Only the four non-zero values are stored in sparse matrix and the positions of
the two two-wide sub-chunks with non-zero values in the eight-wide chunk of a row of matrix A are
indicated by two 2-bit indices in the metadata. `0b0100`, `0b1000`, `0b1001`, `0b1100`, `0b1101`,
`0b1110` are the meaningful values of indices; any other values result in an undefined behavior.

![_images/fp4-metadata-example.png](_images/fp4-metadata-example.png)


Figure 117 Sparse MMA metadata example for `.e2m1` type with `.kind::mxf4` or `.kind::mxf4nvf4`[](#fp4-metadata-example "Permalink to this image")

The sparsity selector indicates the threads which contribute metadata as listed below:

* `m16n8k128`: All threads within a group of four consecutive threads contribute the sparsity metadata.
  Hence, the sparsity selector in this case must be 0. Any other value of sparsity selector results in
  an undefined behavior.