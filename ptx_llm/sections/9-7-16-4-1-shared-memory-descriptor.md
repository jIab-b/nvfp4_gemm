##### 9.7.16.4.1. Shared memory descriptor 

The shared memory descriptor describes the properties of multiplicand matrix in shared
memory including its location in the shared memory of the current *CTA*. It is a 64-bit
value contained in a register with the following layout:

Table 40 Shared memory descriptor layout[](#tcgen05-shared-memory-desc-layout "Permalink to this table")





| Bit-field | Size in bits | Description |
| --- | --- | --- |
| 0-13 | 14 | matrix-descriptor-encode (Matrix start address) |
| 16-29 | 14 | matrix-descriptor-encode ([Leading dimension byte offset relative](#tcgen05-leading-dimension-byte-offset))  OR  matrix-descriptor-encode ([Leading dimension byte address absolute](#tcgen05-leading-dimension-byte-offset)) |
| 32-45 | 14 | matrix-descriptor-encode ([Stride dimension byte offset](#tcgen05-stride-dimension-byte-offset)) |
| 46-48 | 3 | Fixed constant value of 0b001 |
| 49-51 | 3 | Matrix base offset |
| 52 | 1 | Leading dimension stride mode: - 0: byte offset relative - 1: byte address absolute |
| 53-60 | 8 | Fixed constant value of 0xb00000000 |
| 61-63 | 3 | Specifies the swizzling mode to be used: 0. No swizzling 1. 128-Byte with 32B atomic swizzling 2. 128-Byte swizzling 4. 64-Byte swizzling 6. 32-Byte swizzling  Note: Values 3, 5 and 7 are invalid |

where matrix-descriptor-encode(x) = (x & 0x3FFFF) >> 4

The value of base offset is 0 when the repeating pattern of the specified swizzling mode
starts as per shown in [Table 41](#tcgen05-start-addr-swizzle-mode).

Table 41 Starting address of repeating pattern for various swizzling modes[](#tcgen05-start-addr-swizzle-mode "Permalink to this table")




| Swizzling mode | Starting address of the repeating pattern |
| --- | --- |
| 128-Byte swizzle | 1024-Byte boundary |
| 64-Byte swizzle | 512-Byte boundary |
| 32-Byte swizzle | 256-Byte boundary |

Otherwise, the base offset must be a non-zero value, computed using the following formula:
`base offset = (pattern start addr >> 0x7) & 0x7`

The following must be 16-byte aligned:

1. Matrix start address
2. Leading dimension byte offset
3. Stride dimension byte offset

###### 9.7.16.4.1.1. [Target ISA Note](#tcgen05-shared-memory-descriptor-target-isa-note)[](#tcgen05-shared-memory-descriptor-target-isa-note "Permalink to this headline")

* The byte address mode for the leading dimension stride is supported on `sm_103a`.