###### 9.7.16.10.8.2. Sparse tcgen05.mma.sp with .kind::f16 , .kind::f8f6f4 , .kind::mxf8f6f4 , .kind::i8 

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