##### 9.7.13.15.11. Parallel Synchronization and Communication Instructions: mbarrier.expect_tx 

`mbarrier.expect_tx`

Perfoms
[expect-tx](#parallel-synchronization-and-communication-instructions-mbarrier-expect-tx-operation)
operation on the *mbarrier object*.

Syntax

```
mbarrier.expect_tx{.sem}{.scope}{.space}.b64 [addr], txCount;



.sem   = { .relaxed }

.scope = { .cta, .cluster }

.space = { .shared{::cta}, .shared::cluster }
```

Description

A thread executing `mbarrier.expect_tx` performs an [expect-tx](#parallel-synchronization-and-communication-instructions-mbarrier-expect-tx-operation)
operation on the *mbarrier object* at the location specified by the address operand `addr`. The
32-bit unsigned integer operand `txCount` specifies the `expectCount` argument to the
*expect-tx* operation.

If no state space is specified then [Generic Addressing](#generic-addressing) is
used. If the address specified by `addr` does not fall within the address window of
`.shared::cta` or `.shared::cluster` state space then the behavior is undefined.

Supported addressing modes for operand `addr` are as described in [Addresses as Operands](#addresses-as-operands).
Alignment for operand `addr` is as described in the
[Size and alignment of mbarrier object](#parallel-synchronization-and-communication-instructions-mbarrier-size-alignment).

This operation does not provide any memory ordering semantics and thus is a *relaxed* operation.

PTX ISA Notes

Introduced in PTX ISA version 8.0.

Target ISA Notes

Requires `sm_90` or higher.

Examples

```
mbarrier.expect_tx.b64                       [addr], 32;

mbarrier.expect_tx.relaxed.cta.shared.b64    [mbarObj1], 512;

mbarrier.expect_tx.relaxed.cta.shared.b64    [mbarObj2], 512;
```