##### 9.7.16.10.8. Sparse Matrices 

This instruction `tcgen05.mma.sp` can be used when the matrix `A` is a structured
sparse matrix with 50% zeros in each row distributed as per its sparse granularity.

In a *MxNxK* sparse `tcgen05.mma.sp` operation, the matrix `A` of shape *MxK* is
stored in a packed form as *Mx(K/2)* in memory. For each *K-wide* row of matrix `A`,
50% of elements are zeros and the remaining *K/2* non-zero elements are stored in
memory. The metadata specifies the mapping of the *K/2* non-zero elements to the *K*
elements before performing the MMA operation.

Granularity of sparse matrix `A` is defined as the ratio of the number of non-zero
elements in a sub-chunk of the matrix row to the total number of elements in that
sub-chunk where the size of the sub-chunk is shape-specific. The following table lists
the granularity of different `tcgen05.mma.sp` variants:

| .kind of tcgen05.mma | Sparse Granularity |
| --- | --- |
| `.kind::tf32` | 1:2 |
| `.kind::f16` | 2:4 |
| `.kind::f8f6f4` |
| `.kind::mxf8f6f4` |
| `.kind::i8` |
| `.kind::mxf4` | 4:8 (in pairs) |

###### 9.7.16.10.8.1. [Sparse `tcgen05.mma.sp` with `.kind::tf32`](#tcgen05-sparse-matrices-kind-tf32)[](#tcgen05-sparse-matrices-kind-tf32 "Permalink to this headline")

For `.kind::tf32`, matrix `A` is structured sparse at a granularity of `1:2`.
In other words, each chunk of two adjacent elements in a row of matrix `A` has one
zero and one non-zero element. Only the non-zero element is stored in memory and the
4-bit index in the metadata indicates the position of the non-zero element in the
two-wide chunk. The only meaningful values of the index are:

* `0b1110`
* `0b0100`

Rest of the values result in undefined behavior.

![_images/tcgen05-sparse-mma-metadata-tf32.png](_images/tcgen05-sparse-mma-metadata-tf32.png)


Figure 259 Sparse tcgen05.mma metadata example for tf32 kind[](#tcgen05-sparse-mma-metadata-tf32 "Permalink to this image")

###### 9.7.16.10.8.2. [Sparse `tcgen05.mma.sp` with `.kind::f16`, `.kind::f8f6f4`, `.kind::mxf8f6f4`, `.kind::i8`](#tcgen05-sparse-matrices-kind-f16-f8f8f4-mxf8f6f4)[](#tcgen05-sparse-matrices-kind-f16-f8f8f4-mxf8f6f4 "Permalink to this headline")

For the following `.kind` variants of `tcgen05.mma`:

* `.kind::f16`
* `.kind::f8f8f4`
* `.kind::mxf8f6f4`
* `.kind::i8`

matrix `A` is structured sparse at a granularity of `2:4`. In other words, each chunk
of four adjacent elements in a row of matrix `A` has two zero and two non-zero elements.
Only the non-zero elements are stored in memory and the two 2-bit indices in the metadata
indicates the position of the two non-zero elements in the four-wide chunk. The only
meaningful values of the index are:

* `0b0100`
* `0b1000`
* `0b1100`
* `0b1001`
* `0b1101`
* `0b0110`
* `0b1110`

![_images/tcgen05-sparse-mma-metadata-f16-f8f6f4-mxf8f6f4.png](_images/tcgen05-sparse-mma-metadata-f16-f8f6f4-mxf8f6f4.png)


Figure 260 Sparse tcgen05.mma metadata example for f16/f8f6f4/mxf8f6f4 kind[](#tcgen05-sparse-mma-metadata-f16-f8f6f4-mxf8f6f4 "Permalink to this image")

###### 9.7.16.10.8.3. [Sparse `tcgen05.mma.sp` with `.kind::mxf4` and `.kind::mxf4nvf4`](#tcgen05-sparse-matrices-kind-mxf4)[](#tcgen05-sparse-matrices-kind-mxf4 "Permalink to this headline")

For `.kind::mxf4` and `.kind::mxf4nvf4`, matrix `A` is pair-wise structured
sparse at a granularity of `4:8`. In other words, each chunk of eight adjacent
elements in a row of matrix `A` has four zero and four non-zero elements. The
zero and non-zero elements are clustered in sub-chunks of two elements each within
the eight-wide chunk, so each two-wide sub-chunk within the eight-wide chunk must be
all zeros or all non-zeros. Only the four non-zero elements are stored in memory and
the two 2-bit indices in the metadata indicates the position of the two two-wide
sub-chunks with non-zero values in the eight-wide chunk of a row of matrix `A`.
The only meaningful values of the index are:

* `0b0100`
* `0b1000`
* `0b1100`
* `0b1001`
* `0b1101`
* `0b0110`
* `0b1110`

Rest of the values result in undefined behavior.

![_images/tcgen05-sparse-mma-metadata-mxf4.png](_images/tcgen05-sparse-mma-metadata-mxf4.png)


Figure 261 Sparse tcgen05.mma metadata example for mxf4 kind[](#tcgen05-sparse-mma-metadata-mxf4 "Permalink to this image")

###### 9.7.16.10.8.4. [Sparsity selector](#tcgen05-sparse-matrices-sparsity-selector)[](#tcgen05-sparse-matrices-sparsity-selector "Permalink to this headline")

The value of the sparsity selector selects the sub-columns in the Tensor Memory
to form the sparsity metadata matrix, which is used with matrix `A` to form the
multiplicand matrix.

The following shows the sparse metadata matrix layout in Tensor Memory for various MMA variants:

###### 9.7.16.10.8.4.1. [Layout of the Sparsity Metadata Matrix for M = 64 for `.kind::f16`](#tcgen05-sparse-matrices-sparsity-selector-kind-f16-m64)[](#tcgen05-sparse-matrices-sparsity-selector-kind-f16-m64 "Permalink to this headline")

[Figure 262](#tcgen05-sparse-matrices-sparsity-selector-kind-f16-m64-dig) shows which sub-columns gets
selected for different values of Sparsity Selector.

![_images/tcgen05-sparse-matrices-sparsity-selector-kind-f16-m64.png](_images/tcgen05-sparse-matrices-sparsity-selector-kind-f16-m64.png)


Figure 262 Sparsity Metadata Layout for M = 64 for `.kind::f16`[](#tcgen05-sparse-matrices-sparsity-selector-kind-f16-m64-dig "Permalink to this image")

###### 9.7.16.10.8.4.2. [Layout of the Sparsity Metadata Matrix for M = 128 / M = 256 for `.kind::f16`](#tcgen05-sparse-matrices-sparsity-selector-kind-f16-m128-256)[](#tcgen05-sparse-matrices-sparsity-selector-kind-f16-m128-256 "Permalink to this headline")

[Figure 263](#tcgen05-sparse-matrices-sparsity-selector-kind-f16-m128-256-dig) shows which sub-columns gets
selected for different values of Sparsity Selector.

![_images/tcgen05-sparse-matrices-sparsity-selector-kind-f16-m128-256.png](_images/tcgen05-sparse-matrices-sparsity-selector-kind-f16-m128-256.png)


Figure 263 Sparsity Metadata Layout for M = 128 / M = 256 for `.kind::f16`[](#tcgen05-sparse-matrices-sparsity-selector-kind-f16-m128-256-dig "Permalink to this image")

###### 9.7.16.10.8.4.3. [Layout of the Sparsity Metadata Matrix for M = 64 for `.kind::tf32`](#tcgen05-sparse-matrices-sparsity-selector-kind-tf32-m64)[](#tcgen05-sparse-matrices-sparsity-selector-kind-tf32-m64 "Permalink to this headline")

[Figure 264](#tcgen05-sparse-matrices-sparsity-selector-kind-tf32-m64-dig) shows which sub-columns gets
selected for different values of Sparsity Selector.

![_images/tcgen05-sparse-matrices-sparsity-selector-kind-tf32-m64.png](_images/tcgen05-sparse-matrices-sparsity-selector-kind-tf32-m64.png)


Figure 264 Sparsity Metadata Layout for M = 64 for `.kind::tf32`[](#tcgen05-sparse-matrices-sparsity-selector-kind-tf32-m64-dig "Permalink to this image")

###### 9.7.16.10.8.4.4. [Layout of the Sparsity Metadata Matrix for M = 128 / M = 256 for `.kind::tf32`](#tcgen05-sparse-matrices-sparsity-selector-kind-tf32-m128-256)[](#tcgen05-sparse-matrices-sparsity-selector-kind-tf32-m128-256 "Permalink to this headline")

[Figure 265](#tcgen05-sparse-matrices-sparsity-selector-kind-tf32-m128-256-dig) shows which sub-columns gets
selected for different values of Sparsity Selector.

![_images/tcgen05-sparse-matrices-sparsity-selector-kind-tf32-m128-256.png](_images/tcgen05-sparse-matrices-sparsity-selector-kind-tf32-m128-256.png)


Figure 265 Sparsity Metadata Layout for M = 128 / M = 256 for `.kind::tf32`[](#tcgen05-sparse-matrices-sparsity-selector-kind-tf32-m128-256-dig "Permalink to this image")

###### 9.7.16.10.8.4.5. [Layout of the Sparsity Metadata Matrix for M = 64 for `.kind::f8f6f4`, `.kind::mxf8f6f4`, `.kind::i8`, `.kind::mxf4`, `.kind::mxf4nvf4`](#tcgen05-sparse-matrices-sparsity-selector-kind-f8f6f4-mxf8f6f4-m64)[](#tcgen05-sparse-matrices-sparsity-selector-kind-f8f6f4-mxf8f6f4-m64 "Permalink to this headline")

The value of the sparsity selector:

* must be 0 for `.kind::i8` and `.kind::f8f6f4`
* is assumed to be 0 for `.kind::mxf8f6f4`, `.kind::mxf4` and `.kind::mxf4nvf4`

and all of the columns are selected as
shown in [Figure 266](#tcgen05-sparse-matrices-sparsity-selector-kind-f8f6f4-mxf8f6f4-m64-dig)

![_images/tcgen05-sparse-matrices-sparsity-selector-kind-f8f6f4-mxf8f6f4-m64.png](_images/tcgen05-sparse-matrices-sparsity-selector-kind-f8f6f4-mxf8f6f4-m64.png)


Figure 266 Sparsity Metadata Layout for M = 64 for `.kind::f8f6f4`, `.kind::mxf8f6f4`, `.kind::i8`, `.kind::mxf4`, `.kind::mxf4nvf4`[](#tcgen05-sparse-matrices-sparsity-selector-kind-f8f6f4-mxf8f6f4-m64-dig "Permalink to this image")

###### 9.7.16.10.8.4.6. [Layout of the Sparsity Metadata Matrix for M = 128 / M = 256 for `.kind::f8f6f4`, `.kind::mxf8f6f4`, `.kind::i8`, `.kind::mxf4`, `.kind::mxf4nvf4`](#tcgen05-sparse-matrices-sparsity-selector-kind-f8f6f4-mxf8f6f4-m128-256)[](#tcgen05-sparse-matrices-sparsity-selector-kind-f8f6f4-mxf8f6f4-m128-256 "Permalink to this headline")

The value of the sparsity selector:

* must be 0 for `.kind::i8` and `.kind::f8f6f4`
* is assumed to be 0 for `.kind::mxf8f6f4`, `.kind::mxf4` and `.kind::mxf4nvf4`

and all of the columns are selected as
shown in [Figure 267](#tcgen05-sparse-matrices-sparsity-selector-kind-f8f6f4-mxf8f6f4-m128-256-dig)

![_images/tcgen05-sparse-matrices-sparsity-selector-kind-f8f6f4-mxf8f6f4-m128-256.png](_images/tcgen05-sparse-matrices-sparsity-selector-kind-f8f6f4-mxf8f6f4-m128-256.png)


Figure 267 Sparsity Metadata Layout for M = 128 / M = 256 for `.kind::f8f6f4`, `.kind::mxf8f6f4`, `.kind::i8`, `.kind::mxf4`, `.kind::mxf4nvf4`[](#tcgen05-sparse-matrices-sparsity-selector-kind-f8f6f4-mxf8f6f4-m128-256-dig "Permalink to this image")

###### 9.7.16.10.8.5. [Alignment restriction](#tcgen05-sparse-matrices-alignment-restriction)[](#tcgen05-sparse-matrices-alignment-restriction "Permalink to this headline")

The layouts which utilize only half the datapath lanes as specified in
[Data Path Layout Organization](#tcgen05-data-path-layout-organization),
i.e. [Layout F](#tcgen05-data-path-layout-f) and
[Layout C](#tcgen05-data-path-layout-c), must use the same alignment
across matrices A, D and the sparsity metadata matrix.