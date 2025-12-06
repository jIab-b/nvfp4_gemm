##### 9.7.16.4.2. Instruction descriptor 

The instruction descriptor describes the shapes, types and other details of all the matrices
and the matrix-multiplication-and-accumulation operation. It is a 32-bit value in registers
and the exact layout is dependent on the MMA-Kind:

Table 42 Instruction descriptor format for .kind::tf32, .kind::f16, .kind::f8f6f4 and .kind::i8[](#tcgen05-instuction-desc-kind-tf32-f16-f8f6f4 "Permalink to this table")









| Bits | Size  (bits) | Description | Values | | | |
| --- | --- | --- | --- | --- | --- | --- |
| .kind::tf32 | .kind::f16 | .kind::f8f6f4 | .kind::i8 |
| 0-1 | 2 | [Sparsity selector](#tcgen05-sparse-matrices-sparsity-selector), if Sparsity is enabled | 0-3 | | | |
| 2 | 1 | Sparsity | Dense = 0  Sparse = 1 | | | |
| 3 | 1 | Saturate for integer types | 0 (NA) | | | No Saturate = 0 Saturate = 1 |
| 4-5 | 2 | dtype (Matrix D type) | F32 = 1 | F16 = 0 F32 = 1 | | S32 = 2 |
| 6 | 1 | Reserved | 0 | | | |
| 7-9 | 3 | atype (Matrix A type) | TF32 = 2 | F16 = 0  BF16 = 1 | E4M3 = 0 E5M2 = 1 E2M3 = 3 E3M2 = 4 E2M1 = 5 | Unsigned 8b = 0  Signed 8b = 1 |
| 10-12 | 3 | btype (Matrix B type) |
| 13 | 1 | Negate A Matrix | No Negate = 0  Negate = 1 | | | No Negate = 0 |
| 14 | 1 | Negate B Matrix |
| 15 | 1 | Transpose A Matrix | No Transpose = 0  Transpose = 1 | | | |
| 16 | 1 | Transpose B Matrix |
| 17-22 | 6 | N, Dimension of Matrix B (3 LSBs not included) | N >> 3 | | | |
| 23 | 1 | Reserved | 0 | | | |
| 24-28 | 5 | M, Dimension of Matrix A (4 LSBs not included) | M >> 4 | | | |
| 29 | 1 | Reserved | 0 | | | |
| 30-31 | 2 | Maximum shift while attempting B matrix -reuse in `.ws` | no shift = 0 maximum shift of 8 = 1 maximum shift of 16 = 2 maximum shift of 32 = 3 | | | |

Table 43 Instruction descriptor format for .kind::mxf8f6f4[](#tcgen05-instuction-desc-kind-mxf8f6f4 "Permalink to this table")






| Bits | Size  (bits) | Description | Values |
| --- | --- | --- | --- |
| .kind::mxf8f6f4 |
| 0-1 | 2 | Reserved | 0 |
| 2 | 1 | Sparsity | Dense = 0  Sparse = 1 |
| 3 | 1 | Reserved | 0 |
| 4-5 | 2 | [Matrix B Scale Factor Data ID](#tcgen05-mma-scale-factor-b) | 0-3 |
| 6 | 1 | Reserved | 0 |
| 7-9 | 3 | atype (Matrix A type) | E4M3 = 0 E5M2 = 1 E2M3 = 3 E3M2 = 4 E2M1 = 5 |
| 10-12 | 3 | btype (Matrix B type) |
| 13 | 1 | Negate A Matrix | No Negate = 0  Negate = 1 |
| 14 | 1 | Negate B Matrix |
| 15 | 1 | Transpose A Matrix | No Transpose = 0  Transpose = 1 |
| 16 | 1 | Transpose B Matrix |
| 17-22 | 6 | N, Dimension of Matrix B (3 LSBs not included) | N >> 3 |
| 23 | 1 | Scale Matrix Type, for both scale\_A / scale\_B | UE8M0 = 1 |
| 24-26 | 3 | Reserved | 0 |
| 27-28 | 2 | M, Dimension of Matrix A (7 LSBs not included) | M >> 7 |
| 29-30 | 2 | [Matrix A Scale Factor Data ID](#tcgen05-mma-scale-factor-a) | 0-3 |
| 31 | 1 | Reserved | 0 |

Table 44 Instruction descriptor format for .kind::mxf4 and .kind::mxf4nvf4[](#tcgen05-instuction-desc-kind-mxf4-mxf4nvf4 "Permalink to this table")







| Bits | Size  (bits) | Description | Values | |
| --- | --- | --- | --- | --- |
| .kind::mxf4 | .kind::mxf4nvf4 |
| 0-1 | 2 | Reserved | 0 | |
| 2 | 1 | Sparsity | Dense = 0  Sparse = 1 | |
| 3 | 1 | Reserved | 0 | |
| 4-5 | 2 | [Matrix B Scale Factor Data ID](#tcgen05-mma-scale-factor-b) | 0 or 2 | |
| 6 | 1 | Reserved | 0 | |
| 7-9 | 3 | atype (Matrix A type) | E2M1 = 1 | |
| 10-11 | 2 | btype (Matrix B type) |
| 12 | 1 | Reserved | 0 | |
| 13 | 1 | Negate A Matrix | No Negate = 0  Negate = 1 | |
| 14 | 1 | Negate B Matrix |
| 15 | 1 | Transpose A Matrix | No Transpose = 0 | |
| 16 | 1 | Transpose B Matrix |
| 17-22 | 6 | N, Dimension of Matrix B (3 LSBs not included) | N >> 3 | |
| 23 | 1 | Scale Matrix Type, for both scale\_A / scale\_B | UE8M0 = 1 | UE4M3 = 0 |
| 24-26 | 3 | Reserved | 0 | |
| 27-28 | 2 | M, Dimension of Matrix A (7 LSBs not included) | M >> 7 | |
| 29-30 | 2 | [Matrix A Scale Factor Data ID](#tcgen05-mma-scale-factor-a) | 0 or 2 | |
| 31 | 1 | K Dimension | (Dense K=64 / Sparse K=128) = 0  (Dense K=96) = 1 | |