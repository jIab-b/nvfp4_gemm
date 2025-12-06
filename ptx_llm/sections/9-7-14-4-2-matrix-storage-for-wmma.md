##### 9.7.14.4.2. Matrix Storage for WMMA 

Each matrix can be stored in memory with a *row-major* or *column-major* layout. In a *row-major*
format, consecutive elements of each row are stored in contiguous memory locations, and the row is
called the *leading dimension* of the matrix. In a *column-major* format, consecutive elements of
each column are stored in contiguous memory locations and the column is called the *leading
dimension* of the matrix.

Consecutive instances of the *leading dimension* (rows or columns) need not be stored contiguously
in memory. The `wmma.load` and `wmma.store` operations accept an optional argument `stride`
that specifies the offset from the beginning of each row (or column) to the next, in terms of matrix
elements (and not bytes). For example, the matrix being accessed by a `wmma` operation may be a
submatrix from a larger matrix stored in memory. This allows the programmer to compose a
multiply-and-accumulate operation on matrices that are larger than the shapes supported by the
`wmma` operation.

Address Alignment

The starting address of each instance of the leading dimension (row or column) must be aligned
with the size of the corresponding fragment in bytes. Note that the starting address is
determined by the base pointer and the optional `stride`.

Consider the following instruction as an example:

```
wmma.load.a.sync.aligned.row.m16n16k16.f16 {x0,...,x7}, [p], s;
```

* Fragment size in bytes = 32 (eight elements of type `.f16x2`)
* Actual `stride` in bytes = 2 \* `s` (since `stride` is specified in terms of `.f16`
  elements, not bytes)
* For each row of this matrix to be aligned at fragment size the following must be true:

  1. `p` is a multiple of 32.
  2. `2*s` is a multiple of 32.

Default value for stride

The default value of the `stride` is the size of the *leading dimension* of the matrix. For
example, for an `MxK` matrix, the `stride` is `K` for a *row-major* layout and `M` for a
*column-major* layout. In particular, the default strides for the supported matrix shapes are as
follows:

| Shape | A (row) | A (column) | B (row) | B (column) | Accumulator (row) | Accumulator (column) |
| --- | --- | --- | --- | --- | --- | --- |
| 16x16x16 | 16 | 16 | 16 | 16 | 16 | 16 |
| 8x32x16 | 16 | 8 | 32 | 16 | 32 | 8 |
| 32x8x16 | 16 | 32 | 8 | 16 | 8 | 32 |
| 8x8x32 | 32 | 8 | 8 | 32 | 8 | 8 |
| 8x8x128 | 128 | 8 | 8 | 128 | 8 | 8 |
| 16x16x8 | 8 | 16 | 16 | 8 | 16 | 16 |
| 8x8x4 | 4 | 8 | 8 | 4 | 8 | 8 |