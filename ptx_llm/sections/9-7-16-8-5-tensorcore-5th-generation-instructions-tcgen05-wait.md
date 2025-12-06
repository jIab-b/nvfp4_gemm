##### 9.7.16.8.5. Tensorcore 5th Generation Instructions: tcgen05.wait 

`tcgen05.wait`

Waits for the completion of all prior asynchronous `tcgen05.ld` / `tcgen05.st` instructions.

Syntax

```
tcgen05.wait_operation.sync.aligned;



.wait_operation = { .wait::ld, .wait::st }
```

Description

Instruction `tcgen05.wait::st` causes the executing thread to block until all prior
`tcgen05.st` operations issued by the executing thread have completed.

Instruction `tcgen05.wait::ld` causes the executing thread to block until all prior
`tcgen05.ld` operations issued by the executing thread have completed.

The mandatory `.sync` qualifier indicates that `tcgen05.wait_operation` causes the
executing thread to wait until all threads in the warp execute the same `tcgen05.wait_operation`
instruction before resuming execution.

The mandatory `.aligned` qualifier indicates that all threads in the warp must execute the
same `tcgen05.wait_operation` instruction.

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
Example 1:



tcgen05.ld.sync.aligned.32x32b.x2.b32     {r0, r1}, [taddr0];



// Prevents subsequent tcgen05.mma from racing ahead of the tcgen05.ld



tcgen05.wait::ld.sync.aligned;



tcgen05.mma.cta_group::1.kind::f16   [taddr0],  a-desc,  b-desc, idesc, p;



Example 2:



tcgen05.st.sync.aligned.32x32b.x2.b32     [taddr0], {r0, r1};



// Prevents the write to taddr0 in tcgen05.mma from racing ahead of the tcgen05.st



tcgen05.wait::st.sync.aligned;



tcgen05.mma.cta_group::1.kind::f16   [taddr0],  a-desc,  b-desc, idesc, p;
```