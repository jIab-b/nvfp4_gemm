##### 9.7.16.2.1. Matrix Shape 

The matrix multiply and accumulate operations support a limited set of shapes for the operand matrices
`A`, `B` and `D`. The shapes of all three matrix operands are collectively described by the tuple
*MxNxK* where `A` is *MxK* matrix, `B` is a *KxN* matrix, and `D` is a *MxN* matrix.

[Table 39](#tcgen05-kind-shapes) shows matrix shapes that are supported for the specified types for the
`tcgen05.mma` operation.

Table 39 Various combinations of .kind and shapes[](#tcgen05-kind-shapes "Permalink to this table")











| Various Combinations | | | | | | Shapes Supported | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| .kind::\* | Has .ws | CTA Group | Sparsity | dtype | atype/btype |
| `kind::f16` | No `.ws` | 1 | Dense | `.f16` | `.f16` | 64xNxK  128xNxK | N = {8, 16, 24, … 256} steps of 8 | K = 16 |
| `.f32` | `.f16`, `.bf16` |
| Sparse | `.f16` | `.f16` | K = 32 |
| `.f32` | `.f16`, `.bf16` |
| 2 | Dense | `.f16` | `.f16` | 128xNxK  256xNxK | N = {16, 32, … 256} steps of 16 | K = 16 |
| `.f32` | `.f16`, `.bf16` |
| Sparse | `.f16` | `.f16` | K = 32 |
| `.f32` | `.f16`, `.bf16` |
| `.ws` | 1 | Dense | `.f16` | `.f16` | 32xNxK  64xNxK  128xNxK | N = {64, 128, 256} | K = 16 |
| `.f32` | `.f16`, `.bf16` |
| Sparse | `.f16` | `.f16` | N = {64, 128} | K = 32 |
| `.f32` | `.f16`, `.bf16` |
| 2 | Either | `.f16` | `.f16` | Invalid | | |
| `.f32` | `.f16`, `.bf16` |
| `.kind::tf32` | No `.ws` | 1 | Dense | `.f32` | `.tf32` | 64xNxK  128xNxK | N = {8, 16, 24, … 256} steps of 8 | K = 8 |
| Sparse | K = 16 |
| 2 | Dense | 128xNxK  256xNxK | N = {16, 32, … 256} steps of 16 | K = 8 |
| Sparse | K = 16 |
| `.ws` | 1 | Dense | 32xNxK 64xNxK 128xNxK | N = {64, 128, 256} | K = 8 |
| Sparse | N = {64, 128} | K = 16 |
| 2 | Dense | Invalid | | |
| Sparse |
| `.kind::f8f6f4` | No `.ws` | 1 | Dense | `.f32`  `.f16` | `.e4m3`,  `.e5m2`,  `.e2m3`,  `.e3m2`,  `.e2m1` | 64xNxK  128xNxK | N = {8, 16, … 256} steps of 8 | K = 32 |
| Sparse | K = 64 |
| 2 | Dense | 128xNxK  256xNxK | N = {16, 32, … 256} steps of 16 | K = 32 |
| Sparse | K = 64 |
| `.ws` | 1 | Dense | 32xNxK 64xNxK 128xNxK | N = {64, 128, 256} | K = 32 |
| Sparse | N = {64, 128} | K = 64 |
| 2 | Dense | Invalid | | |
| Sparse |
| `.kind::mxf8f6f4` | No `.ws` | 1 | Dense | `.f32` | `.e4m3`,  `.e5m2`,  `.e2m3`,  `.e3m2`,  `.e2m1`  X  (Scale)  `.ue8m0` | 128xNxK | N = {8, 16, … 256} steps of 8 | K = 32 |
| Sparse | K = 64 |
| 2 | Dense | 128xNxK  256xNxK | N = {16, 32, … 256} steps of 16 | K = 32 |
|
| Sparse | 256xNxK | K = 64 |
| `.ws` | 1 | Dense | Invalid | | |
| Sparse |
| 2 | Dense |
| Sparse |
| `.kind::i8` | No `.ws` | 1 | Dense | `.s32` | `.s8`, `.u8` | 64xNxK  128xNxK | N = {8, 16, 24, 32, 48, … 256}  steps of 16 after N > 32 | K = 32 |
| Sparse | K = 64 |
| 2 | Dense | 128xNxK  256xNxK | N = {32, 64, … 256} steps of 32 | K = 32 |
| Sparse | K = 64 |
| `.ws` | 1 | Dense | 32xNxK 64xNxK 128xNxK | N = {64, 128, 256} | K = 32 |
| Sparse | N = {64, 128} | K = 64 |
| 2 | Dense | Invalid | | |
| Sparse |
| `.kind::mxf4` | No `.ws` | 1 | Dense | `.f32` | `.e2m1`  X  (Scale)  `.ue8m0` | 128xNxK | N = {8, 16, … 256} steps of 8 | K = 64 |
| Sparse | K = 128 |
| 2 | Dense | 128xNxK 256xNxK 256xNxK1 | N = {16, 32, … 256} steps of 16 | K = 64  K1 = 96 |
|
| Sparse | 256xNxK | K = 128 |
| `.ws` | 1 / 2 | Either | Invalid | | |
| `.kind::mxf4nvf4` | No `.ws` | 1 | Dense | `.f32` | `.e2m1`  X  (Scale)  `.ue8m0`,  `.ue4m3` | 128xNxK | N = {8, 16, … 256} steps of 8 | K = 64 |
| Sparse | K = 128 |
| 2 | Dense | 128xNxK 256xNxK 256xNxK1 | N = {16, 32, … 256} steps of 16 | K = 64  K1 = 96 |
|
| Sparse | 256xNxK | K = 128 |
| `.ws` | 1 / 2 | Either | Invalid | | |

###### 9.7.16.2.1.1. [Target ISA Note](#tcgen05-matrix-shape-target-isa-note)[](#tcgen05-matrix-shape-target-isa-note "Permalink to this headline")

* K = 96 is only supported for target architecture `sm_103a`.