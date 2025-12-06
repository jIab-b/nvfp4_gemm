##### 9.7.16.3.1. Leading Dimension Stride: relative offset or absolute address 

There are two modes of Leading Dimension Strides as described below.
Bit #52 in the [Shared memory descriptor](#tcgen05-shared-memory-descriptor) is used to distinguish between two modes.

###### 9.7.16.3.1.1. [Relative offset mode](#tcgen05-leading-dimension-byte-offset-relative-offset)[](#tcgen05-leading-dimension-byte-offset-relative-offset "Permalink to this headline")

In this mode, the leading dimension stride is specified as a relative byte offset between the
columns as explained in the below table.
The leading dimension stride can either be specified as a relative offset between the columns
or as an absolute byte address of next buffer. The leading dimension stride is defined
differently for transposed and non-transposed matrices. The leading dimension stride is defined
as follows for matrices whose element types are normalized to 128-bits:

| Major-ness | Definition |
| --- | --- |
| K-Major | * No-Swizzling: the stride from the first column to the second column   of the 8x2 tile in the 128-bit element type normalized matrix. * Swizzled layouts: not used, assumed to be 1. |
| MN-Major | * Interleave: stride from the first 8 columns to the next 8 columns. * Swizzled layouts: stride from the first (swizzle-byte-size/16) rows   to the next (swizzle-byte-size/16) rows. |

###### 9.7.16.3.1.2. [Absolute address mode for K dimension being 48B](#tcgen05-leading-dimension-byte-offset-absolute-address)[](#tcgen05-leading-dimension-byte-offset-absolute-address "Permalink to this headline")

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