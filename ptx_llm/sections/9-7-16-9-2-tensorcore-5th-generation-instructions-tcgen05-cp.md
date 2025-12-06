##### 9.7.16.9.2. Tensorcore 5th Generation Instructions: tcgen05.cp 

`tcgen05.cp`

Initiates an asynchronous copy operation from shared memory to the [Tensor Memory](#tensor-memory).

Syntax

```
tcgen05.cp.cta_group.shape{.multicast}{.dst_fmt.src_fmt} [taddr], s-desc;



.cta_group = { .cta_group::1, .cta_group::2 }

.src_fmt   = { .b6x16_p32 , .b4x16_p64 }

.dst_fmt   = { .b8x16 }

.shape     = { .128x256b, .4x256b, .128x128b, .64x128b**, .32x128b*** }

.multicast = { .warpx2::02_13** , .warpx2::01_23**, .warpx4*** }
```

Description

Instruction `tcgen05.cp` initiates an asynchronous copy operation from shared memory to the
location specified by the address operand `taddr` in the [Tensor Memory](#tensor-memory).

The 64-bit register operand `s-desc` is the matrix descriptor which represents the source
matrix in the shared memory that needs to be copied. The format of the matrix descriptor is
described in [Matrix Descriptors](#tcgen05-matrix-descriptors).

The `.shape` qualifier indicates the dimension of data to be copied as described in the
[Data Movement Shape](#tcgen05-data-movement-shape).

Qualifier `.cta_group` specifies the number of CTAs whose [Tensor Memory](#tensor-memory) is
accessed when a single thread of a single CTA executes the `tcgen05.cp` instruction.
When `.cta_group::1` is specified, the data is copied into the [Tensor Memory](#tensor-memory)
of the current CTA. When `.cta_group::2` is specified, the data is copied into the
[Tensor Memory](#tensor-memory) of both the current and the [peer CTAs](#tcgen05-peer-cta).

All `tcgen05` instructions within a kernel must specify the same value for the `.cta_group`
qualifier.

When the qualifiers `.dst_fmt` and `.src_fmt` are specified, the data is decompressed
from the source format `.src_fmt` in the shared memory to the destination format
`.dst_fmt` in [Tensor Memory](#tensor-memory) by the copy operation. The details of source
and the destination formats as specified in the section
[Optional Decompression](#tcgen05-optional-decompression).

Some of the `.shape` qualifiers require certain `.multicast` qualifiers.

1. `.64x128b` requires `.warpx2::02_13` or `.warpx2::01_23`
2. `.32x128b` requires `.warpx4`

When the `.multicast` qualifier is specified as either `.warpx2::02_13` or
`.warpx2::01_23` then the data being copied is multicasted into warp pairs and each
warp in the warp pair receive half of the data. Warp pairs are formed as follows:

1. `.warpx2::02_13` : warps 0 and 2 form a pair; warps 1 and 3 form a pair.
2. `.warpx2::01_23` : warps 0 and 1 form a pair; warps 2 and 3 form a pair.

When the `.multicast` modifier is specified as `.warpx4` then the data being
copied is multicasted into all 4 warps.

PTX ISA Notes

Introduced in PTX ISA version 8.6.

Target ISA Notes

Supported on following architectures:

* `sm_100a`
* `sm_101a` (Renamed to `sm_110a` from PTX ISA version 9.0)
* And is supported on following family-specific architectures from PTX ISA version 8.8:

  + `sm_100f` or higher in the same family
  + `sm_101f` or higher in the same family (Renamed to `sm_110f` from PTX ISA version 9.0)
* `sm_110f` or higher in the same family

Examples

```
tcgen05.cp.cta_group::1.128x256b                 [taddr0], sdesc0;

tcgen05.cp.cta_group::2.128x128b.b8x16.b6x16_p32 [taddr1], sdesc1;

tcgen05.cp.cta_group::1.64x128b.warpx2::02_13    [taddr2], sdesc2;
```