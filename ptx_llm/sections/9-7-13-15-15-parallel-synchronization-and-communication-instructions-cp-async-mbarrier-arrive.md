##### 9.7.13.15.15. Parallel Synchronization and Communication Instructions: cp.async.mbarrier.arrive 

`cp.async.mbarrier.arrive`

Makes the *mbarrier object* track all prior [cp.async](#data-movement-and-conversion-instructions-cp-async)
operations initiated by the
executing thread.

Syntax

```
cp.async.mbarrier.arrive{.noinc}{.shared{::cta}}.b64 [addr];
```

Description

Causes an [arrive-on operation](#parallel-synchronization-and-communication-instructions-mbarrier-arrive-on) to be
triggered by the system on the *mbarrier object* upon the completion of all prior [cp.async](#data-movement-and-conversion-instructions-cp-async)
operations initiated by the
executing thread. The *mbarrier object* is at the location specified by the operand `addr`. The
[arrive-on operation](#parallel-synchronization-and-communication-instructions-mbarrier-arrive-on) is
asynchronous to execution of `cp.async.mbarrier.arrive`.

When `.noinc` modifier is not specified, the pending count of the mbarrier object is incremented
by 1 prior to the asynchronous [arrive-on operation](#parallel-synchronization-and-communication-instructions-mbarrier-arrive-on). This
results in a zero-net change for the pending count from the asynchronous [arrive-on](#parallel-synchronization-and-communication-instructions-mbarrier-arrive-on) operation
during the current phase. The pending count of the *mbarrier object* after the increment should not
exceed the limit as mentioned in
[Contents of the mbarrier object](#parallel-synchronization-and-communication-instructions-mbarrier-contents). Otherwise,
the behavior is undefined.

When the `.noinc` modifier is specified, the increment to the pending count of the *mbarrier
object* is not performed. Hence the decrement of the pending count done by the asynchronous
[arrive-on operation](#parallel-synchronization-and-communication-instructions-mbarrier-arrive-on) must be
accounted for in the initialization of the *mbarrier object*.

If no state space is specified then [Generic Addressing](#generic-addressing) is
used. If the address specified by `addr` does not fall within the address window of
`.shared::cta` state space then the behavior is undefined.

Supported addressing modes for operand `addr` is as described in [Addresses as Operands](#addresses-as-operands).
Alignment for operand `addr` is as described in the
[Size and alignment of mbarrier object](#parallel-synchronization-and-communication-instructions-mbarrier-size-alignment).

PTX ISA Notes

Introduced in PTX ISA version 7.0.

Support for sub-qualifier `::cta` on `.shared` introduced in PTX ISA version 7.8.

Target ISA Notes

Requires `sm_80` or higher.

Examples

```
// Example 1: no .noinc

mbarrier.init.shared.b64 [shMem], threadCount;

....

cp.async.ca.shared.global [shard1], [gbl1], 4;

cp.async.cg.shared.global [shard2], [gbl2], 16;

....

// Absence of .noinc accounts for arrive-on from completion of prior cp.async operations.

// So mbarrier.init must only account for arrive-on from mbarrier.arrive.

cp.async.mbarrier.arrive.shared.b64 [shMem];

....

mbarrier.arrive.shared.b64 state, [shMem];



waitLoop:

mbarrier.test_wait.shared.b64 p, [shMem], state;

@!p bra waitLoop;







// Example 2: with .noinc



// Tracks arrive-on from mbarrier.arrive and cp.async.mbarrier.arrive.



// All threads participating in the mbarrier perform cp.async

mov.b32 copyOperationCnt, threadCount;



// 3 arrive-on operations will be triggered per-thread

mul.lo.u32 copyArrivalCnt, copyOperationCnt, 3;



add.u32 totalCount, threadCount, copyArrivalCnt;



mbarrier.init.shared.b64 [shMem], totalCount;

....

cp.async.ca.shared.global [shard1], [gbl1], 4;

cp.async.cg.shared.global [shard2], [gbl2], 16;

...

// Presence of .noinc requires mbarrier initalization to have accounted for arrive-on from cp.async

cp.async.mbarrier.arrive.noinc.shared.b64 [shMem]; // 1st instance

....

cp.async.ca.shared.global [shard3], [gbl3], 4;

cp.async.ca.shared.global [shard4], [gbl4], 16;

cp.async.mbarrier.arrive.noinc.shared::cta.b64 [shMem]; // 2nd instance

....

cp.async.ca.shared.global [shard5], [gbl5], 4;

cp.async.cg.shared.global [shard6], [gbl6], 16;

cp.async.mbarrier.arrive.noinc.shared.b64 [shMem]; // 3rd and last instance

....

mbarrier.arrive.shared.b64 state, [shMem];



waitLoop:

mbarrier.test_wait.shared.b64 p, [shMem], state;

@!p bra waitLoop;
```