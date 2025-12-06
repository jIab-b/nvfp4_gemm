###### 9.7.16.3.1.1. Relative offset mode 

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