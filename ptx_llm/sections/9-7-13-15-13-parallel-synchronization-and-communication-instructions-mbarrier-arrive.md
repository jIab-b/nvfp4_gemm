##### 9.7.13.15.13. Parallel Synchronization and Communication Instructions: mbarrier.arrive 

`mbarrier.arrive`

Performs [arrive-on operation](#parallel-synchronization-and-communication-instructions-mbarrier-arrive-on) on the
*mbarrier object*.

Syntax

```
mbarrier.arrive{.sem}{.scope}{.shared{::cta}}.b64           state, [addr]{, count};

mbarrier.arrive{.sem}{.scope}{.shared::cluster}.b64         _, [addr] {,count}

mbarrier.arrive.expect_tx{.sem}{.scope}{.shared{::cta}}.b64 state, [addr], txCount;

mbarrier.arrive.expect_tx{.sem}{.scope}{.shared::cluster}.b64   _, [addr], txCount;

mbarrier.arrive.noComplete{.release}{.cta}{.shared{::cta}}.b64  state, [addr], count;



.sem   = { .release, .relaxed }

.scope = { .cta, .cluster }
```

Description

A thread executing `mbarrier.arrive` performs an [arrive-on](#parallel-synchronization-and-communication-instructions-mbarrier-arrive-on) operation
on the *mbarrier object* at the location specified by the address operand `addr`. The 32-bit
unsigned integer operand `count` specifies the *count* argument to the [arrive-on](#parallel-synchronization-and-communication-instructions-mbarrier-arrive-on)
operation.

If no state space is specified then [Generic Addressing](#generic-addressing) is
used. If the address specified by `addr` does not fall within the address window of
`.shared::cta` state space then the behavior is undefined.

Supported addressing modes for operand `addr` is as described in [Addresses as Operands](#addresses-as-operands).
Alignment for operand `addr` is as described in the
[Size and alignment of mbarrier object](#parallel-synchronization-and-communication-instructions-mbarrier-size-alignment).

The optional qualifier `.expect_tx` specifies that an [expect-tx](#parallel-synchronization-and-communication-instructions-mbarrier-expect-tx-operation)
operation is performed prior to the [arrive-on](#parallel-synchronization-and-communication-instructions-mbarrier-arrive-on)
operation. The 32-bit unsigned integer operand `txCount` specifies the *expectCount* argument to
the *expect-tx* operation. When both qualifiers `.arrive` and `.expect_tx` are specified, then
the count argument of the *arrive-on* operation is assumed to be 1.

A `mbarrier.arrive` operation with `.noComplete` qualifier must not cause the `mbarrier` to
complete its current phase, otherwise the behavior is undefined.

The value of the operand `count` must be in the range as specified in
[Contents of the mbarrier object](#parallel-synchronization-and-communication-instructions-mbarrier-contents).

Note: for `sm_8x`, when the argument `count` is specified, the modifier `.noComplete` is
required.

`mbarrier.arrive` operation on an *mbarrier object* located in `.shared::cta` returns an opaque
64-bit register capturing the phase of the *mbarrier object* prior to the [arrive-on operation](#parallel-synchronization-and-communication-instructions-mbarrier-arrive-on) in the
destination operand `state.` Contents of the `state` operand are implementation
specific. Optionally, sink symbol `'_'` can be used for the `state` argument.

`mbarrier.arrive` operation on an *mbarrier object* located in `.shared::cluster` but not in
`.shared::cta` cannot return a value. Sink symbol ‘\_’ is mandatory for the destination operand for
such cases.

The optional `.sem` qualifier specifies a memory synchronizing effect as described in the
[Memory Consistency Model](#memory-consistency-model). If the `.sem` qualifier is absent,
`.release` is assumed by default.

The `.relaxed` qualifier does not provide any memory ordering semantics and visibility
guarantees.

The optional `.scope` qualifier indicates the set of threads that directly observe the memory
synchronizing effect of this operation, as described in the [Memory Consistency Model](#memory-consistency-model).
If the `.scope` qualifier is not specified then it
defaults to `.cta`. In contrast, the `.shared::<scope>` indicates the state space where the
mbarrier resides.

PTX ISA Notes

Introduced in PTX ISA version 7.0.

Support for sink symbol ‘\_’ as the destination operand is introduced in PTX ISA version 7.1.

Support for sub-qualifier `::cta` on `.shared` introduced in PTX ISA version 7.8.

Support for `count` argument without the modifier `.noComplete` introduced in PTX ISA version
7.8.

Support for sub-qualifier `::cluster` introduced in PTX ISA version 8.0.

Support for qualifier `.expect_tx` is introduced in PTX ISA version 8.0.

Support for `.scope` and `.sem` qualifiers introduced in PTX ISA version 8.0

Support for `.relaxed` qualifier introduced in PTX ISA version 8.6.

Target ISA Notes

Requires `sm_80` or higher.

Support for `count` argument without the modifier `.noComplete` requires `sm_90` or higher.

Qualifier `.expect_tx` requires `sm_90` or higher.

Sub-qualifier `::cluster` requires `sm_90` or higher.

Support for `.cluster` scope requires `sm_90` or higher.

Examples

```
.reg .b32 cnt, remoteAddr32, remoteCTAId, addr32;

.reg .b64 %r<5>, addr, remoteAddr64;

.shared .b64 shMem, shMem2;



cvta.shared.u64            addr, shMem2;

mov.b32                    addr32, shMem2;

mapa.shared::cluster.u32   remoteAddr32, addr32, remoteCTAId;

mapa.u64                   remoteAddr64, addr,   remoteCTAId;



cvta.shared.u64          addr, shMem2;



mbarrier.arrive.shared.b64                       %r0, [shMem];

mbarrier.arrive.shared::cta.b64                  %r0, [shMem2];

mbarrier.arrive.release.cta.shared::cluster.b64  _, [remoteAddr32];

mbarrier.arrive.release.cluster.b64              _, [remoteAddr64], cnt;

mbarrier.arrive.expect_tx.release.cluster.b64    _, [remoteAddr64], tx_count;

mbarrier.arrive.noComplete.b64                   %r1, [addr], 2;

mbarrier.arrive.relaxed.cta.b64                  %r2, [addr], 4;

mbarrier.arrive.b64                              %r2, [addr], cnt;
```