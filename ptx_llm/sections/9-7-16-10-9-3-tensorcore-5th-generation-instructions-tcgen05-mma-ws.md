###### 9.7.16.10.9.3. TensorCore 5th Generation Instructions: tcgen05.mma.ws 

`tcgen05.mma.ws`

Perform the 5th generation of weight stationary convolution matrix multiply and accumulate
operation.

Syntax

```
// 1. Floating-point type without block scaling:



tcgen05.mma.ws.cta_group::1.kind{.collector_usage}    [d-tmem],  a-desc,  b-desc,  idesc,

                                                      enable-input-d {, zero-column-mask-desc };



tcgen05.mma.ws.cta_group::1.kind{.collector_usage}    [d-tmem], [a-tmem], b-desc, idesc,

                                                      enable-input-d {, zero-column-mask-desc };



.kind = { .kind::f16, .kind::tf32, .kind::f8f6f4 }



----------------------------------------------------------------------------------



// 2. Integer type:



tcgen05.mma.ws.cta_group::1.kind::i8{.collector_usage} [d-tmem],  a-desc,  b-desc, idesc,

                                                       enable-input-d {, zero-column-mask-desc};



tcgen05.mma.ws.cta_group::1.kind::i8{.collector_usage} [d-tmem], [a-tmem], b-desc, idesc,

                                                       enable-input-d {, zero-column-mask-desc};



.collector_usage = { .collector::buffer::op }

::buffer = { ::b0, ::b1, ::b2, ::b3 }

::op   = { ::fill, ::use, ::lastuse, ::discard}
```

Description

Instruction `tcgen05.mma.ws` is an asynchronous instruction which initiates an *MxNxK*
matrix multiply and accumulate operation,
`D = A*B+D`
where the `A` matrix is *MxK*, the `B` matrix is *KxN*, and the `D` matrix is *MxN*.

The operation of the form
`D = A*B`
is issued when the input predicate argument `enable-input-d` is false.

The 32-bit register operand `idesc` is the instruction descriptor as described in
[Instruction descriptor](#tcgen05-instruction-descriptor), specifies the shapes, exact
types, sparsity and other details of the input matrices, output matrix and the matrix
multiply and accumulate operation.

The qualifier `.cta_group::1` specifies that the matrix multiply and accumulate operation
is performed on the [Tensor Memory](#tensor-memory) of the executing thread’s CTA only.

All `tcgen05` instructions within a kernel must specify the same value for the `.cta_group`
qualifier.

The instruction `tcgen05.mma.ws` has single thread semantics, unlike the collective
instructions `mma.sync` or `wgmma.mma_async`. So, a single thread issuing the
`tcgen05.mma.ws` will result in the initiation of the whole matrix multiply and accumulate
operation. Refer to the section [Issue Granularity](#tcgen05-issue-granularity).

The qualifier `.kind` specifies the general kind of the element types of the multiplicand
matrices. The exact types of the elements of the input and output matrices for each MMA-kind
are specified in the [Instruction descriptor](#tcgen05-instruction-descriptor).

The address operand `d-tmem` specifies the address of the destination and the accumulation
matrix `D` in the [Tensor Memory](#tensor-memory). The address operand `a-tmem`
specifies the address of the matrix `A` in the [Tensor Memory](#tensor-memory). The
64-bit register operand `a-desc` and `b-desc` are the matrix descriptors which represent
the matrices `A` and `B` in shared memory respectively. The format of the matrix descriptor
is described in [Matrix Descriptors](#tcgen05-matrix-descriptors).

The optional operand `zero-column-mask-desc` is a 64-bit register which specifies the
[Zero-Column Mask Descriptor](#tcgen05-zero-column-mask-descriptor). The zero-column
mask descriptor is used to generate a mask that specifies which columns of `B` matrix
will have zero value for the matrix multiply and accumulate operation regardless of the
values present in the shared memory.

The qualifier `.collector_usage` specifies the usage of collector buffer for Matrix `B`.
Following collector buffer operations can be specified:

| .collector\_usage | Semantics |
| --- | --- |
| `.collector::bN::fill` | Specifies that the `B` matrix read from the memory should be filled in collector buffer #N. |
| `.collector::bN::use` | Specifies that the `B` matrix can be read from the collector buffer #N. This requires a previous fill to the collector buffer #N to be still valid. |
| `.collector::bN::lastuse` | Specifies that the `B` matrix can be read from the collector buffer #N after which the contents of the collector buffer #N can be discarded. This requires a previous fill to the collector buffer #N to be valid till the collector buffer #N is read. |
| `.collector::bN::discard` | Specifies that the contents of the collector buffer #N can be discarded. |

If no `.collector_usage` qualifier is specified, then it defaults to `.collector::b0::discard`.

PTX ISA Notes

Introduced in PTX ISA version 8.6.

Target ISA Notes

Supported on following architectures:

* `sm_100a`
* `sm_101a` (Renamed to `sm_110a` from PTX ISA version 9.0)
* And is supported on following family-specific architectures from PTX ISA version 8.8 except `.kind::i8`:

  + `sm_100f` or higher in the same family
  + `sm_101f` or higher in the same family (Renamed to `sm_110f` from PTX ISA version 9.0)
* `sm_110f` or higher in the same family

Qualifier `.kind::i8` is supported on following architectures:

* `sm_100a`
* `sm_101a` (Renamed to `sm_110a` from PTX ISA version 9.0)
* `sm_110a`

Examples

```
tcgen05.mma.ws.cta_group::1.kind::i8.collector::b2:use [taddr2], [taddr1], bdesc, idesc, p;

tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [mbarObj0];



loop:

mbarrier.try_wait.parity.b64 p, [mbarObj0], 0;

@!p bra loop;
```