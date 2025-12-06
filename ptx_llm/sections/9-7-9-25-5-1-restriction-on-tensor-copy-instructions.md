###### 9.7.9.25.5.1. Restriction on Tensor Copy instructions 

Following are the restrictions on the types `.b4x16`, `.b4x16_p64`, `.b6x16_p32` and
`.b6p2x16`:

1. `cp.reduce.async.bulk` doesn’t support the types `.b4x16`, `.b4x16_p64`, `.b6x16_p32`
   and `.b6p2x16`.
2. `cp.async.bulk.tensor` with the direction `.global.shared::cta` doesn’t support the
   type `.b4x16_p64`.
3. `cp.async.bulk.tensor` with the direction `.shared::cluster.global` doesn’t support
   the sub-byte types on `sm_120a`.
4. OOB-NaN fill mode doesn’t support the types `.b4x16`, `.b4x16_p64`, `.b6x16_p32`
   and `.b6p2x16`.
5. Box-Size[0] must be exactly:

   1. 96B for `b6x16_p32` and `.b6p2x16`.
   2. 64B for `b4x16_p64`.
6. Tensor-Size[0] must be a multiple of:

   1. 96B for `b6x16_p32` and `.b6p2x16`.
   2. 64B for `b4x16_p64`.
7. For `.b4x16_p64`, `.b6x16_p32` and `.b6p2x16`, the first coordinate in the tensorCoords
   argument vector must be a multiple of 128.
8. For `.b4x16_p64`, `.b6x16_p32` and `.b6p2x16`, the global memory address must be 32B aligned.
   Additionally, tensor stride in every dimension must be 32B aligned.
9. `.b4x16_p64`, `.b6x16_p32` and `.b6p2x16` supports the following swizzling modes:

   1. None.
   2. 128B (With all potential swizzle atomicity values except: 32B with 8B flip)

Following are the restrictions on the 96B swizzle mode:

1. The `.swizzle_atomicity` must be 16B.
2. The `.interleave_layout` must not be set.
3. Box-Size[0] must be less than or equal to 96B.
4. The type must not be among following: `.b4x16_p64`, `.b6x16_p32` and `.b6p2x16`.
5. The `.load_mode` must not be set to `.im2col::w::128`.

Following are the restrictions on the `.global.shared::cta` direction:

1. Starting co-ordinates for Bounding Box (`tensorCoords`) must be non-negative.
2. The bounding box along the D, W and H dimensions must stay within the tensor boundaries.
   This implies:

   1. Bounding-Box Lower-Corner must be non-negative.
   2. Bounding-Box Upper-Corner must be non-positive.

Following are the restrictions for `sm_120a`:

1. `cp.async.bulk.tensor` with the direction `.shared::cluster.global` doesn’t support:

   1. the sub-byte types
   2. the qualifier `.swizzle_atomicity`

Following are the restrictions for `sm_103a` while using type `.b6p2x16` on
`cp.async.bulk.tensor` with the direction `.global.shared::cta`:

1. Box-Size[0] must be exactly either of 48B or 96B.
2. The global memory address must be 16B aligned.
3. Tensor Stride in every dimension must be 16B aligned.
4. The first coordinate in the tensorCoords argument vector must be a multiple of 64.
5. Tensor-Size[0] must be a multiple of 48B.
6. The following swizzle modes are supported:

   1. None.
   2. 128B (With all potential swizzle atomicity values except: 32B with 8B flip)
   3. 64B swizzle with 16B swizzle atomicity