##### 9.7.13.15.14. Parallel Synchronization and Communication Instructions: mbarrier.arrive_drop 

`mbarrier.arrive_drop`

Decrements the expected count of the *mbarrier object* and performs [arrive-on operation](#parallel-synchronization-and-communication-instructions-mbarrier-arrive-on).

Syntax

```
mbarrier.arrive_drop{.sem}{.scope}{.shared{::cta}}.b64 state,           [addr]{, count};

mbarrier.arrive_drop{.sem}{.scope}{.shared::cluster}.b64           _,   [addr] {,count};

mbarrier.arrive_drop.expect_tx{.shared{::cta}}{.sem}{.scope}.b64 state, [addr], tx_count;

mbarrier.arrive_drop.expect_tx{.shared::cluster}{.sem}{.scope}.b64   _, [addr], tx_count;

mbarrier.arrive_drop.noComplete{.release}{.cta}{.shared{::cta}}.b64 state,  [addr], count;



.sem   = { .release, .relaxed }

.scope = { .cta, .cluster }
```

Description

A thread executing `mbarrier.arrive_drop` on the *mbarrier object* at the location specified by
the address operand `addr` performs the following steps:

* Decrements the expected arrival count of the *mbarrier object* by the value specified by the
  32-bit integer operand `count`. If `count` operand is not specified, it defaults to 1.
* Performs an [arrive-on operation](#parallel-synchronization-and-communication-instructions-mbarrier-arrive-on) on the
  *mbarrier object*. The operand `count` specifies the *count* argument to the [arrive-on operation](#parallel-synchronization-and-communication-instructions-mbarrier-arrive-on).

The decrement done in the expected arrivals count of the *mbarrier object* will be for all the
subsequent phases of the *mbarrier object*.

If no state space is specified then [Generic Addressing](#generic-addressing) is
used. If the address specified by `addr` does not fall within the address window of
`.shared::cta` or `.shared::cluster` state space then the behavior is undefined.

Supported addressing modes for operand `addr` is as described in [Addresses as Operands](#addresses-as-operands).
Alignment for operand `addr` is as described in the
[Size and alignment of mbarrier object](#parallel-synchronization-and-communication-instructions-mbarrier-size-alignment).

The optional qualifier `.expect_tx` specifies that an [expect-tx](#parallel-synchronization-and-communication-instructions-mbarrier-expect-tx-operation)
operation is performed prior to the [arrive-on](#parallel-synchronization-and-communication-instructions-mbarrier-arrive-on)
operation. The 32-bit unsigned integer operand `txCount` specifies the *expectCount* argument to
the *expect-tx* operation. When both qualifiers `.arrive` and `.expect_tx` are specified, then
the count argument of the *arrive-on* operation is assumed to be 1.

`mbarrier.arrive_drop` operation with `.release` qualifier forms the *release* pattern as
described in the Memory Consistency Model and synchronizes with the *acquire* patterns.

The optional `.sem` qualifier specifies a memory synchronizing effect as described in the
[Memory Consistency Model](#memory-consistency-model). If the `.sem` qualifier is absent,
`.release` is assumed by default. The `.relaxed` qualifier does not provide any memory
ordering semantics and visibility guarantees.

The optional `.scope` qualifier indicates the set of threads that an `mbarrier.arrive_drop`
instruction can directly synchronize. If the `.scope` qualifier is not specified then it defaults
to `.cta`. In contrast, the `.shared::<scope>` indicates the state space where the mbarrier
resides.

A `mbarrier.arrive_drop` with `.noComplete` qualifier must not complete the `mbarrier,`
otherwise the behavior is undefined.

The value of the operand `count` must be in the range as specified in
[Contents of the mbarrier object](#parallel-synchronization-and-communication-instructions-mbarrier-contents).

Note: for `sm_8x`, when the argument `count` is specified, the modifier `.noComplete` is
required.

A thread that wants to either exit or opt out of participating in the [arrive-on operation](#parallel-synchronization-and-communication-instructions-mbarrier-arrive-on) can use
`mbarrier.arrive_drop` to drop itself from the `mbarrier`.

`mbarrier.arrive_drop` operation on an *mbarrier object* located in `.shared::cta` returns an
opaque 64-bit register capturing the phase of the *mbarrier object* prior to the [arrive-on
operation](#parallel-synchronization-and-communication-instructions-mbarrier-arrive-on)
in the destination operand `state`. Contents of the returned state are implementation
specific. Optionally, sink symbol `'_'` can be used for the `state` argument.

`mbarrier.arrive_drop` operation on an *mbarrier* object located in `.shared::cluster` but not
in `.shared::cta` cannot return a value. Sink symbol ‘\_’ is mandatory for the destination operand
for such cases.

PTX ISA Notes

Introduced in PTX ISA version 7.0.

Support for sub-qualifier `::cta` on `.shared` introduced in PTX ISA version 7.8.

Support for `count` argument without the modifier `.noComplete` introduced in PTX ISA version
7.8.

Support for qualifier `.expect_tx` is introduced in PTX ISA version 8.0.

Support for sub-qualifier `::cluster` introduced in PTX ISA version 8.0.

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
.reg .b32 cnt;

.reg .b64 %r1;

.shared .b64 shMem;



// Example 1

@p mbarrier.arrive_drop.shared.b64 _, [shMem];

@p exit;

@p2 mbarrier.arrive_drop.noComplete.shared.b64 _, [shMem], %a;

@p2 exit;

..

@!p mbarrier.arrive.shared.b64   %r1, [shMem];

@!p mbarrier.test_wait.shared.b64  q, [shMem], %r1;



// Example 2

mbarrier.arrive_drop.shared::cluster.b64 _, [addr];

mbarrier.arrive_drop.shared::cta.release.cluster.b64     _, [addr], cnt;



// Example 3

mbarrier.arrive_drop.expect_tx.shared::cta.relaxed.cluster.b64 state, [addr], tx_count;
```