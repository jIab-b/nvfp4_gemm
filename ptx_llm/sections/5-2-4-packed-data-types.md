### 5.2.4. Packed Data Types 

Certain PTX instructions operate on two or more sets of inputs in parallel, and produce two or more
outputs. Such instructions can use the data stored in a packed format. PTX supports packing two or
four values of the same scalar data type into a single, larger value. The packed value is considered
as a value of a *packed data type*. In this section we describe the packed data types supported in PTX.

#### 5.2.4.1. [Packed Floating Point Data Types](#packed-floating-point-data-types)[](#packed-floating-point-data-types "Permalink to this headline")

PTX supports various variants of packed floating point data types. Out of them, only `.f16x2` is
supported as a fundamental type, while others cannot be used as fundamental types - they are
supported as instruction types on certain instructions. When using an instruction with such
non-fundamental types, the operand data variables must be of bit type of appropriate size.
For example, all of the operand variables must be of type `.b32` for an instruction with
instruction type as `.bf16x2`.
[Table 9](#operand-types-for-packed-floating-point-instruction-type) described various variants
of packed floating point data types in PTX.

Table 9 Operand types for packed floating point instruction type.[](#operand-types-for-packed-floating-point-instruction-type "Permalink to this table")






| Packed floating point type | Number of elements contained in a packed format | Type of each element | Register variable type to be used in the declaration |
| --- | --- | --- | --- |
| `.f16x2` | Two | `.f16` | `.f16x2` or `.b32` |
| `.f32x2` | `.f32` | `.b64` |
| `.bf16x2` | `.bf16` | `.b32` |
| `.e4m3x2` | `.e4m3` | `.b16` |
| `.e5m2x2` | `.e5m2` |
| `.e2m3x2` | `.e2m3` |
| `.e3m2x2` | `.e3m2` |
| `.ue8m0x2` | `.ue8m0` |
| `.e2m1x2` | `.e2m1` | `.b8` |
| `.e4m3x4` | Four | `.e4m3` | `.b32` |
| `.e5m2x4` | `.e5m2` |
| `.e2m3x4` | `.e2m3` |
| `.e3m2x4` | `.e3m2` |
| `.e2m1x4` | `.e2m1` |

#### 5.2.4.2. [Packed Integer Data Types](#packed-integer-data-types)[](#packed-integer-data-types "Permalink to this headline")

PTX supports two variants of packed integer data types: `.u16x2` and `.s16x2`. The packed data
type consists of two `.u16` or `.s16` values. A register variable containing `.u16x2` or
`.s16x2` data must be declared with `.b32` type. Packed integer data types cannot be used as
fundamental types. They are supported as instruction types on certain instructions.