###### 9.7.16.3.1.2. Absolute address mode for K dimension being 48B 

The `tcgen05.mma` instruction with *K-dimension* of 48B would overflow the 128B
shared memory boundary if the data is packed contiguously.

In this case, the absolute address mode can be used to break up the data in the
shared memory into two chunks such that both these chunks are laid out within
the aligned 128-byte address boundary.
The leading dimension absolute address can point to the second data chunk in the shared memory.

###### 9.7.16.3.1.2.1. [Restrictions on the Leading Dimension Absolute Address Stride](#tcgen05-leading-dimension-byte-offset-absolute-address-restriction)[](#tcgen05-leading-dimension-byte-offset-absolute-address-restriction "Permalink to this headline")

Following are the restrictions on the absolute address stride mode:

1. Only 128B swizzle (with 16B atomicity) is supported.
2. Only K-Major mode is supported. That is, the transpose bits(bits #15 and #16) in
   [Instruction descriptor](#tcgen05-instruction-descriptor) must be 0.
3. The matrix base offset must be 0.