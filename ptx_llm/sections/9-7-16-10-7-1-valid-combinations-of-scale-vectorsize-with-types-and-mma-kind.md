###### 9.7.16.10.7.1. Valid combinations of scale_vectorsize with types and MMA-Kind 

The shape of *scale\_A* and *scale\_B* matrices depend on the `.scale_vectorsize` as shown in
[Table 54](#tcgen05-mma-scale-valid-comb).

Table 54 Valid combinations of scale\_vectorsize and shapes[](#tcgen05-mma-scale-valid-comb "Permalink to this table")







| .scale\_vectorsize | .kind::\* | K | Shape of scale\_A | Shape of scale\_B |
| --- | --- | --- | --- | --- |
| `.scale_vec::1X` | `.kind::mxf8f6f4` | All supported values of K | M x 1 | 1 x N |
| `.scale_vec::2X` | `.kind::mxf4`, `.kind::mxf4nvf4` | All supported values of K | M x 2 | 2 x N |
| `.scale_vec::4X` | `.kind::mxf4nvf4` | All supported values of K | M x 4 | 4 x N |
| `.block16` | `.kind::mxf4nvf4` | K = 96 | M x 6 | 6 x N |
| All supported values of K except 96 | M x 4 | 4 x N |
| `.block32` | `.kind::mxf4`, `.kind::mxf4nvf4` | K = 96 | M x 3 | 3 x N |
| All supported values of K except 96 | M x 2 | 2 x N |
| `.kind::mxf8f6f4` | All supported values of K | M x 1 | 1 x N |

The valid combination of the exact element types and the `.scale_vectorsize` are listed in
[Table 55](#tcgen05-mma-scale-valid-comb-detail).

Table 55 Valid combinations of scale\_vectorsize with types and MMA-Kind[](#tcgen05-mma-scale-valid-comb-detail "Permalink to this table")






| .kind::\* | Element Data Type | Scale Data Type | .scale\_vectorsize |
| --- | --- | --- | --- |
| `.kind::mxf8f6f4` | E4M3, E5M2, E2M3 E3M2, E2M1 | UE8M0 | `.scale_vec::1X` / `.block32` |
| `.kind::mxf4` | E2M1 | UE8M0 | `.scale_vec::2X` / `.block32` |
| `.kind::mxf4nvf4` | E2M1 | UE8M0 | `.scale_vec::2X` / `.block32`, `.scale_vec::4X` / `.block16` |
| E2M1 | UE4M3 | `.scale_vec::4X` / `.block16` |

New `.blockN` qualifiers are aliases for `.scale_vec::NX` qualifiers as:

* `.block32` is alias for `.scale_vec::1X` or `.scale_vec::2X`
  based on `.kind` and K dimension
* `.block16` is alias for `.scale_vec::4X`