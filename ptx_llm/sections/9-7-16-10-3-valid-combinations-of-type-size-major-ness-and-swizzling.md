##### 9.7.16.10.3. Valid Combinations of Type-Size, Major-ness and Swizzling 

Table 52 Valid Combinations of Type-Size, Major-ness and Swizzling[](#tcgen05-matrices-valid-type-size-majorness-swizzle "Permalink to this table")






| Type-Size | Major-ness | Matrix | Supported Swizzle |
| --- | --- | --- | --- |
| 4-bit, 6-bit, 8-bit, 16-bit, 32-bit | Row | A | All swizzling modes |
| Column | B |
| 8-bit  16-bit | Column (transpose) | A | All except 128B swizzling with 32B atomicity |
| Row (transpose) | B |
| 32-bit | Column (transpose) | A | Only 128B swizzling with 32B atomicity |
| Row (transpose) | B |