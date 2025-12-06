###### 9.7.9.25.4.1. Data Movement and Conversion Instructions: cp.async.bulk 

`cp.async.bulk`

Initiates an asynchronous copy operation from one state space to another.

Syntax

```
// global -> shared::cta

cp.async.bulk.dst.src.completion_mechanism{.level::cache_hint}

                      [dstMem], [srcMem], size, [mbar] {, cache-policy}



.dst =                  { .shared::cta }

.src =                  { .global }

.completion_mechanism = { .mbarrier::complete_tx::bytes }

.level::cache_hint =    { .L2::cache_hint }





// global -> shared::cluster

cp.async.bulk.dst.src.completion_mechanism{.multicast}{.level::cache_hint}

                      [dstMem], [srcMem], size, [mbar] {, ctaMask} {, cache-policy}



.dst =                  { .shared::cluster }

.src =                  { .global }

.completion_mechanism = { .mbarrier::complete_tx::bytes }

.level::cache_hint =    { .L2::cache_hint }

.multicast =            { .multicast::cluster  }





// shared::cta -> shared::cluster

cp.async.bulk.dst.src.completion_mechanism [dstMem], [srcMem], size, [mbar]



.dst =                  { .shared::cluster }

.src =                  { .shared::cta }

.completion_mechanism = { .mbarrier::complete_tx::bytes }





// shared::cta -> global

cp.async.bulk.dst.src.completion_mechanism{.level::cache_hint}{.cp_mask}

                      [dstMem], [srcMem], size {, cache-policy} {, byteMask}



.dst =                  { .global }

.src =                  { .shared::cta }

.completion_mechanism = { .bulk_group }

.level::cache_hint =    { .L2::cache_hint }
```

Description

`cp.async.bulk` is a non-blocking instruction which initiates an asynchronous bulk-copy operation
from the location specified by source address operand `srcMem` to the location specified by
destination address operand `dstMem`.

The direction of bulk-copy is from the state space specified by the `.src` modifier to the state
space specified by the `.dst` modifiers.

The 32-bit operand `size` specifies the amount of memory to be copied, in terms of number of
bytes. `size` must be a multiple of 16. If the value is not a multiple of 16, then the behavior is
undefined. The memory range `[dstMem, dstMem + size - 1]` must not overflow the destination memory
space and the memory range `[srcMem, srcMem + size - 1]` must not overflow the source memory
space. Otherwise, the behavior is undefined. The addresses `dstMem` and `srcMem` must be aligned
to 16 bytes.

When the destination of the copy is `.shared::cta` the destination address has to be in the shared
memory of the executing CTA within the cluster, otherwise the behavior is undefined.

When the source of the copy is `.shared::cta` and the destination is `.shared::cluster`, the
destination has to be in the shared memory of a different CTA within the cluster.

The modifier `.completion_mechanism` specifies the completion mechanism that is supported on the
instruction variant. The completion mechanisms that are supported for different variants are
summarized in the following table:

| .completion-mechanism | `.dst` | `.src` | Completion mechanism | |
| --- | --- | --- | --- | --- |
| Needed for completion of entire Async operation | optionally can be used for the completion of - Reading data from the source - Reading from the tensormap, if applicable |
| `.mbarrier::...` | `.shared::cta` | `.global` | mbarrier based | *Bulk async-group* based |
| `.shared::cluster` | `.global` |
| `.shared::cluster` | `.shared::cta` |
| `.bulk_group` | `.global` | `.shared::cta` | *Bulk async-group* based |

The modifier `.mbarrier::complete_tx::bytes` specifies that the `cp.async.bulk` variant uses
mbarrier based completion mechanism. The [complete-tx](#parallel-synchronization-and-communication-instructions-mbarrier-complete-tx-operation)
operation, with `completeCount` argument equal to amount of data copied in bytes, will be
performed on the mbarrier object specified by the operand `mbar`.

The modifier `.bulk_group` specifies that the `cp.async.bulk` variant uses *bulk async-group*
based completion mechanism.

The optional modifier `.multicast::cluster` allows copying of data from global memory to shared
memory of multiple CTAs in the cluster. Operand `ctaMask` specifies the destination CTAs in the
cluster such that each bit position in the 16-bit `ctaMask` operand corresponds to the `%ctaid`
of the destination CTA. The source data is multicast to the same CTA-relative offset as `dstMem`
in the shared memory of each destination CTA. The mbarrier signal is also multicast to the same
CTA-relative offset as `mbar` in the shared memory of the destination CTA.

When the optional argument `cache-policy` is specified, the qualifier `.level::cache_hint` is
required. The 64-bit operand `cache-policy` specifies the cache eviction policy that may be used
during the memory access.

`cache-policy` is a hint to the cache subsystem and may not always be respected. It is treated as
a performance hint only, and does not change the memory consistency behavior of the program. The
qualifier `.level::cache_hint` is only supported when at least one of the `.src` or `.dst`
statespaces is `.global` state space.

When the optional qualifier `.cp_mask` is specified, the argument `byteMask` is required.
The i-th bit in the 16-bit wide `byteMask` operand specifies whether the i-th byte of each 16-byte
wide chunk of source data is copied to the destination. If the bit is set, the byte is copied.

The copy operation in `cp.async.bulk` is treated as a weak memory operation and the
[complete-tx](#parallel-synchronization-and-communication-instructions-mbarrier-complete-tx-operation)
operation on the mbarrier has `.release` semantics at the `.cluster` scope as described in the
[Memory Consistency Model](#memory-consistency-model).

Notes

`.multicast::cluster` qualifier is optimized for target architecture `sm_90a`/`sm_100f`/`sm_100a`/
`sm_103f`/`sm_103a`/`sm_110f`/`sm_110a` and may have substantially reduced performance on other
targets and hence `.multicast::cluster` is advised to be used with `.target` `sm_90a`/`sm_100f`/
`sm_100a`/`sm_103f`/`sm_103a`/`sm_110f`/`sm_110a`.

PTX ISA Notes

Introduced in PTX ISA version 8.0.

Support for `.shared::cta` as destination state space is introduced in PTX ISA version 8.6.

Support for `.cp_mask` qualifier introduced in PTX ISA version 8.6.

Target ISA Notes

Requires `sm_90` or higher.

`.multicast::cluster` qualifier advised to be used with `.target` `sm_90a` or `sm_100f` or
`sm_100a` or `sm_103f` or `sm_103a` or `sm_110f` or `sm_110a`.

Support for `.cp_mask` qualifier requires `sm_100` or higher.

Examples

```
// .global -> .shared::cta (strictly non-remote):

cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [dstMem], [srcMem], size, [mbar];



cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint

                                             [dstMem], [srcMem], size, [mbar], cache-policy;



// .global -> .shared::cluster:

cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [dstMem], [srcMem], size, [mbar];



cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster

                                             [dstMem], [srcMem], size, [mbar], ctaMask;



cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint

                                             [dstMem], [srcMem], size, [mbar], cache-policy;





// .shared::cta -> .shared::cluster (strictly remote):

cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [dstMem], [srcMem], size, [mbar];



// .shared::cta -> .global:

cp.async.bulk.global.shared::cta.bulk_group [dstMem], [srcMem], size;



cp.async.bulk.global.shared::cta.bulk_group.L2::cache_hint} [dstMem], [srcMem], size, cache-policy;



// .shared::cta -> .global with .cp_mask:

cp.async.bulk.global.shared::cta.bulk_group.L2::cache_hint.cp_mask [dstMem], [srcMem], size, cache-policy, byteMask;
```