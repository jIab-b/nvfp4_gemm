##### 9.7.16.4.3. Zero-Column Mask Descriptor 

The zero-column mask descriptor is used to generate a mask that specifies which columns of
`B` matrix will have zero value for the MMA operation regardless of the values present in
the shared memory. The total size of the generated mask is N-bits.

A 0-bit in the mask specifies that values of the corresponding column in matrix `B` should
be used for the MMA operation. A 1-bit in the mask specifies 0s must be used for the entire
column for the MMA operation.

The zero-column mask descriptor is a 64-bit value in registers with the following layout:

Table 45 Zero-Column Mask descriptor layout[](#tcgen05-zero-column-mask-desc "Permalink to this table")






| Bits | Size (bits) | Field Name | Description |
| --- | --- | --- | --- |
| 0-7 | 8 | Start Count 0 (sc0) | Specifies the LSBs that must be skipped  for sub-mask mask-i |
| 8-15 | 8 | Start Count 1 (sc1) |
| 16-23 | 8 | Start Count 2 (sc2) |
| 24-31 | 8 | Start Count 3 (sc3) |
| 32 | 1 | First Span 0 (fs0) | Specifies the starting value for  sub-mask mask-i |
| 33 | 1 | First Span 1 (fs1) |
| 34 | 1 | First Span 2 (fs2) |
| 35 | 1 | First Span 3 (fs3) |
| 36-38 | 3 | Reserved |  |
| 39 | 1 | Non-Zero Mask | Value 0 indicates generated mask will have all 0s Value 1 indicates the mask has to be generated |
| 40-47 | 8 | Skip Span | (Count of consecutive columns where B matrix is used) - 1 |
| 48-55 | 8 | Use Span | (Count of consecutive columns where 0s ar used) - 1 |
| 56-61 | 6 | Column Shift | Shifts column by specified amount. Thus allows MMA on non-0 starting column. Max shift amount = 16 for M=32 Max shift amount = 32 otherwise |

The zero-column mask is made up of one or more sub-mask depending on M, as shown in the table:

| M | Zero-Column Mask breakup | Sub-masks | First Span used | Start Column used |
| --- | --- | --- | --- | --- |
| 128 | Single sub-mask of size N-bits | mask0 | fs0 | sc0 |
| 64 | Two sub-masks, each with size of N/2 bits | mask0, mask1 | fs0, fs1 | sc0, sc1 |
| 32 | Four sub-masks, each with size of N/4 bits | mask0, mask1 mask2, mask3 | fs0, fs1, fs2, fs3 | sc0, sc1, sc2, sc3 |

The following table shows the coverage of the sub-masks across N-dimension:

| Sub-mask | M | | |
| --- | --- | --- | --- |
| 128 | 64 | 32 |
| mask0 | Columns [0, N-1] | Columns [0, N/2-1] | Columns [0, N/4-1] |
| mask1 | – | Columns [N/2, N-1] | Columns [N/4, N/2-1] |
| mask2 | – | – | Columns [N/2, (N/4\*3)-1] |
| mask3 | – | – | Columns [(N/4\*3), N-1] |

The following examples shows zero-column mask descriptor and their corresponding mask generated:

1. Example 1: M = 128

   Input zero-column mask descriptor:

   | Start count | First span | Non-Zero Mask | Skip Span | Use Span | Shift |
   | --- | --- | --- | --- | --- | --- |
   | {0, 0, 0, 0} | {0, 0, 0, 0} | 0 | 4 | 3 | 0 |

   Output zero-column mask: 0x0.

   As Non-Zero Mask field is 0, the mask is 0x0. All the columns of the matrix `B` will be used
   for the MMA operation.
2. Example 2: M = 128

   Input zero-column mask descriptor:

   | Start count | First span | Non-Zero Mask | Skip Span | Use Span | Shift |
   | --- | --- | --- | --- | --- | --- |
   | {-, -, -, 0} | {-, -, -, 0} | 1 | 2 | 3 | 0 |

   Output mask0: 0b … 111 0000 111 0000 (size = N)
3. Example 3: M = 64

   Input zero-column mask descriptor:

   | Start count {.., sc1, sc0} | First span {.., fs1, fs0} | Non-Zero Mask | Skip Span | Use Span | Shift |
   | --- | --- | --- | --- | --- | --- |
   | {-, -, 0, 0} | {-, -, 0, 1} | 1 | 2 | 3 | 0 |

   Output mask0: 0b … 111 0000 111 0000 111

   Output masl1: 0b … 0000 111 0000 111 0000
4. Example 4: M = 32

   Input zero-column mask descriptor:

   | Start count {sc3, sc2, sc1, sc0} | First span {fs3, fs2, fs1, fs0} | Non-Zero Mask | Skip Span | Use Span | Shift |
   | --- | --- | --- | --- | --- | --- |
   | {1, 2, 1, 0} | {0, 0, 1, 1} | 1 | 2 | 3 | 2 |

   Output mask0: 0b … 0000 111 0000 111

   Output mask1: 0b … 0000 111 0000 11

   Output mask2: 0b … 111 0000 111 00

   Output mask3: 0b … 111 0000 111 000

   If N = 128 then `B` Matrix with columns from 2 to 129 will be used for the MMA operation,
   due to the shift of 2.