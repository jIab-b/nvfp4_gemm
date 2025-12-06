##### 9.7.13.15.16. Parallel Synchronization and Communication Instructions: mbarrier.test_wait / mbarrier.try_wait 

`mbarrier.test_wait`, `mbarrier.try_wait`

Checks whether the *mbarrier object* has completed the phase.

Syntax

```
mbarrier.test_wait{.sem}{.scope}{.shared{::cta}}.b64        waitComplete, [addr], state;

mbarrier.test_wait.parity{.sem}{.scope}{.shared{::cta}}.b64 waitComplete, [addr], phaseParity;



mbarrier.try_wait{.sem}{.scope}{.shared{::cta}}.b64         waitComplete, [addr], state

                                                               {, suspendTimeHint};



mbarrier.try_wait.parity{.sem}{.scope}{.shared{::cta}}.b64  waitComplete, [addr], phaseParity

                                                               {, suspendTimeHint};



.sem   = { .acquire, .relaxed }

.scope = { .cta, .cluster }
```

Description

The *test\_wait* and *try\_wait* operations test for the completion of the current or the immediately
preceding phase of an *mbarrier object* at the location specified by the operand `addr`.

`mbarrier.test_wait` is a non-blocking instruction which tests for the completion of the phase.

`mbarrier.try_wait` is a potentially blocking instruction which tests for the completion of the
phase. If the phase is not complete, the executing thread may be suspended. Suspended thread resumes
execution when the specified phase completes OR before the phase completes following a
system-dependent time limit. The optional 32-bit unsigned integer operand `suspendTimeHint`
specifies the time limit, in nanoseconds, that may be used for the time limit instead of the
system-dependent limit.

`mbarrier.test_wait` and `mbarrier.try_wait` test for completion of the phase :

* Specified by the operand `state`, which was returned by an `mbarrier.arrive` instruction on
  the same *mbarrier object* during the current or the immediately preceding phase. Or
* Indicated by the operand `phaseParity`, which is the integer parity of either the current phase
  or the immediately preceding phase of the *mbarrier object*.

The `.parity` variant of the instructions test for the completion of the phase indicated by the
operand `phaseParity`, which is the integer parity of either the current phase or the immediately
preceding phase of the *mbarrier object*. An even phase has integer parity 0 and an odd phase has
integer parity of 1. So the valid values of `phaseParity` operand are 0 and 1.

Note: the use of the `.parity` variants of the instructions requires tracking the phase of an
*mbarrier object* throughout its lifetime.

The *test\_wait* and *try\_wait* operations are valid only for :

* the current incomplete phase, for which `waitComplete` returns `False`.
* the immediately preceding phase, for which `waitComplete` returns `True`.

If no state space is specified then [Generic Addressing](#generic-addressing) is
used. If the address specified by `addr` does not fall within the address window of
`.shared::cta` state space then the behavior is undefined.

Supported addressing modes for operand `addr` is as described in [Addresses as Operands](#addresses-as-operands).
Alignment for operand `addr` is as described in the
[Size and alignment of mbarrier object](#parallel-synchronization-and-communication-instructions-mbarrier-size-alignment).

When `mbarrier.test_wait` and `mbarrier.try_wait` operations with `.acquire` qualifier
returns `True`, they form the *acquire* pattern as described in the
[Memory Consistency Model](#memory-consistency-model).

The optional `.sem` qualifier specifies a memory synchronizing effect as described in the
[Memory Consistency Model](#memory-consistency-model). If the `.sem` qualifier is absent,
`.acquire` is assumed by default. The `.relaxed` qualifier does not provide any memory
ordering semantics and visibility guarantees.

The optional `.scope` qualifier indicates the set of threads that the `mbarrier.test_wait` and
`mbarrier.try_wait` instructions can directly synchronize. If the `.scope` qualifier is not
specified then it defaults to `.cta`. In contrast, the `.shared::<scope>` indicates the state
space where the mbarrier resides.

The following ordering of memory operations hold for the executing thread when
`mbarrier.test_wait` or `mbarrier.try_wait` having acquire semantics returns `True` :

1. All memory accesses (except [async operations](#data-movement-and-conversion-instructions-cp-async)) requested prior, in program
   order, to `mbarrier.arrive` having release semantics during the completed phase by
   the participating threads of the CTA are performed and are visible to the executing thread.
2. All [cp.async](#data-movement-and-conversion-instructions-cp-async) operations
   requested prior, in program order, to `cp.async.mbarrier.arrive` during the completed phase by
   the participating threads of the CTA are performed and made visible to the executing thread.
3. All `cp.async.bulk` asynchronous operations using the same *mbarrier object* requested prior,
   in program order, to `mbarrier.arrive` having release semantics during the completed
   phase by the participating threads of the CTA are performed and made visible to the executing thread.
4. All memory accesses requested after the `mbarrier.test_wait` or `mbarrier.try_wait`, in
   program order, are not performed and not visible to memory accesses performed prior to
   `mbarrier.arrive` having release semantics, in program order, by other threads
   participating in the `mbarrier`.
5. There is no ordering and visibility guarantee for memory accesses requested by the thread after
   `mbarrier.arrive` having release semantics and prior to `mbarrier.test_wait`,
   in program order.

PTX ISA Notes

`mbarrier.test_wait` introduced in PTX ISA version 7.0.

Modifier `.parity` is introduced in PTX ISA version 7.1.

`mbarrier.try_wait` introduced in PTX ISA version 7.8.

Support for sub-qualifier `::cta` on `.shared` introduced in PTX ISA version 7.8.

Support for `.scope` and `.sem` qualifiers introduced in PTX ISA version 8.0

Support for `.relaxed` qualifier introduced in PTX ISA version 8.6.

Target ISA Notes

`mbarrier.test_wait` requires `sm_80` or higher.

`mbarrier.try_wait` requires `sm_90` or higher.

Support for `.cluster` scope requires `sm_90` or higher.

Examples

```
// Example 1a, thread synchronization with test_wait:



.reg .b64 %r1;

.shared .b64 shMem;



mbarrier.init.shared.b64 [shMem], N;  // N threads participating in the mbarrier.

...

mbarrier.arrive.shared.b64  %r1, [shMem]; // N threads executing mbarrier.arrive



// computation not requiring mbarrier synchronization...



waitLoop:

mbarrier.test_wait.shared.b64    complete, [shMem], %r1;

@!complete nanosleep.u32 20;

@!complete bra waitLoop;



// Example 1b, thread synchronization with try_wait :



.reg .b64 %r1;

.shared .b64 shMem;



mbarrier.init.shared.b64 [shMem], N;  // N threads participating in the mbarrier.

...

mbarrier.arrive.shared.b64  %r1, [shMem]; // N threads executing mbarrier.arrive



// computation not requiring mbarrier synchronization...



waitLoop:

mbarrier.try_wait.relaxed.cluster.shared.b64    complete, [shMem], %r1;

@!complete bra waitLoop;





// Example 2, thread synchronization using phase parity :



.reg .b32 i, parArg;

.reg .b64 %r1;

.shared .b64 shMem;



mov.b32 i, 0;

mbarrier.init.shared.b64 [shMem], N;  // N threads participating in the mbarrier.

...

loopStart :                           // One phase per loop iteration

    ...

    mbarrier.arrive.shared.b64  %r1, [shMem]; // N threads

    ...

    and.b32 parArg, i, 1;

    waitLoop:

    mbarrier.test_wait.parity.shared.b64  complete, [shMem], parArg;

    @!complete nanosleep.u32 20;

    @!complete bra waitLoop;

    ...

    add.u32 i, i, 1;

    setp.lt.u32 p, i, IterMax;

@p bra loopStart;





// Example 3, Asynchronous copy completion waiting :



.reg .b64 state;

.shared .b64 shMem2;

.shared .b64 shard1, shard2;

.global .b64 gbl1, gbl2;



mbarrier.init.shared.b64 [shMem2], threadCount;

...

cp.async.ca.shared.global [shard1], [gbl1], 4;

cp.async.cg.shared.global [shard2], [gbl2], 16;



// Absence of .noinc accounts for arrive-on from prior cp.async operation

cp.async.mbarrier.arrive.shared.b64 [shMem2];

...

mbarrier.arrive.shared.b64 state, [shMem2];



waitLoop:

mbarrier.test_wait.shared::cta.b64 p, [shMem2], state;

@!p bra waitLoop;



// Example 4, Synchronizing the CTA0 threads with cluster threads

.reg .b64 %r1, addr, remAddr;

.shared .b64 shMem;



cvta.shared.u64          addr, shMem;

mapa.u64                 remAddr, addr, 0;     // CTA0's shMem instance



// One thread from CTA0 executing the below initialization operation

@p0 mbarrier.init.shared::cta.b64 [shMem], N;  // N = no of cluster threads



barrier.cluster.arrive;

barrier.cluster.wait;



// Entire cluster executing the below arrive operation

mbarrier.arrive.release.cluster.b64              _, [remAddr];



// computation not requiring mbarrier synchronization ...



// Only CTA0 threads executing the below wait operation

waitLoop:

mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64  complete, [shMem], 0;

@!complete bra waitLoop;
```