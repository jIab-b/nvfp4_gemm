###### 9.7.16.10.8.3. Sparse tcgen05.mma.sp with .kind::mxf4 and .kind::mxf4nvf4 

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