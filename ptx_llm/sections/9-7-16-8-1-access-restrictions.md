##### 9.7.16.8.1. Access restrictions 

Not all threads of the CTA can access the entire Tensor Memory via the `tcgen05.ld` and
`tcgen05.st` operations.

The Tensor Memory of a CTA is divided into 4 equal chunks such that each warp of a warpgroup
in the CTA can access a chunk of the Tensor Memory. All the columns of the Tensor Memory can
be accessed by all the four warps of a warpgroup. A lane of the Tensor Memory can be accessed
by a single warp in the warpgroup. The following table describes the access restriction.

| ID of the warp within the warpgroup | Accessible Lanes |
| --- | --- |
| 0 | 0-31 |
| 1 | 32-63 |
| 2 | 64-95 |
| 3 | 96-127 |