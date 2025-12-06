###### 9.7.16.10.8.1. Sparse tcgen05.mma.sp with .kind::tf32 

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