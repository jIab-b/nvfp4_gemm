##### 9.7.9.25.3. Data Movement and Conversion Instructions: Non-bulk copy 

###### 9.7.9.25.3.1. [Data Movement and Conversion Instructions: `cp.async`](#data-movement-and-conversion-instructions-cp-async)[](#data-movement-and-conversion-instructions-cp-async "Permalink to this headline")

`cp.async`

Initiates an asynchronous copy operation from one state space to another.

Syntax

```
cp.async.ca.shared{::cta}.global{.level::cache_hint}{.level::prefetch_size}

                         [dst], [src], cp-size{, src-size}{, cache-policy} ;

cp.async.cg.shared{::cta}.global{.level::cache_hint}{.level::prefetch_size}

                         [dst], [src], 16{, src-size}{, cache-policy} ;

cp.async.ca.shared{::cta}.global{.level::cache_hint}{.level::prefetch_size}

                         [dst], [src], cp-size{, ignore-src}{, cache-policy} ;

cp.async.cg.shared{::cta}.global{.level::cache_hint}{.level::prefetch_size}

                         [dst], [src], 16{, ignore-src}{, cache-policy} ;



.level::cache_hint =     { .L2::cache_hint }

.level::prefetch_size =  { .L2::64B, .L2::128B, .L2::256B }

cp-size =                { 4, 8, 16 }
```

Description

`cp.async` is a non-blocking instruction which initiates an asynchronous copy operation of data
from the location specified by source address operand `src` to the location specified by
destination address operand `dst`. Operand `src` specifies a location in the global state space
and `dst` specifies a location in the shared state space.

Operand `cp-size` is an integer constant which specifies the size of data in bytes to be copied to
the destination `dst`. `cp-size` can only be 4, 8 and 16.

Instruction `cp.async` allows optionally specifying a 32-bit integer operand `src-size`. Operand
`src-size` represents the size of the data in bytes to be copied from `src` to `dst` and must
be less than `cp-size`. In such case, remaining bytes in destination `dst` are filled with
zeros. Specifying `src-size` larger than `cp-size` results in undefined behavior.

The optional and non-immediate predicate argument `ignore-src` specifies whether the data from the
source location `src` should be ignored completely. If the source data is ignored then zeros will
be copied to destination `dst`. If the argument `ignore-src` is not specified then it defaults
to `False`.

Supported alignment requirements and addressing modes for operand `src` and `dst` are described
in [Addresses as Operands](#addresses-as-operands).

The mandatory `.async` qualifier indicates that the `cp` instruction will initiate the memory
copy operation asynchronously and control will return to the executing thread before the copy
operation is complete. The executing thread can then use
[async-group based completion mechanism](#data-movement-and-conversion-instructions-asynchronous-copy-completion-mechanisms-async-group)
or the [mbarrier based completion mechanism](#data-movement-and-conversion-instructions-asynchronous-copy-completion-mechanisms-mbarrier)
to wait for completion of the asynchronous copy operation.
No other synchronization mechanism guarantees the completion of the asynchronous
copy operations.

There is no ordering guarantee between two `cp.async` operations if they are not explicitly
synchronized using `cp.async.wait_all` or `cp.async.wait_group` or [mbarrier instructions](#parallel-synchronization-and-communication-instructions-mbarrier).

As described in [Cache Operators](#cache-operators), the `.cg` qualifier indicates
caching of data only at global level cache L2 and not at L1 whereas `.ca` qualifier indicates
caching of data at all levels including L1 cache. Cache operator are treated as performance hints
only.

`cp.async` is treated as a weak memory operation in the [Memory Consistency Model](#memory-consistency-model).

The `.level::prefetch_size` qualifier is a hint to fetch additional data of the specified size
into the respective cache level.The sub-qualifier `prefetch_size` can be set to either of `64B`,
`128B`, `256B` thereby allowing the prefetch size to be 64 Bytes, 128 Bytes or 256 Bytes
respectively.

The qualifier `.level::prefetch_size` may only be used with `.global` state space and with
generic addressing where the address points to `.global` state space. If the generic address does
not fall within the address window of the global memory, then the prefetching behavior is undefined.

The `.level::prefetch_size` qualifier is treated as a performance hint only.

When the optional argument `cache-policy` is specified, the qualifier `.level::cache_hint` is
required. The 64-bit operand `cache-policy` specifies the cache eviction policy that may be used
during the memory access.

The qualifier `.level::cache_hint` is only supported for `.global` state space and for generic
addressing where the address points to the `.global` state space.

`cache-policy` is a hint to the cache subsystem and may not always be respected. It is treated as
a performance hint only, and does not change the memory consistency behavior of the program.

PTX ISA Notes

Introduced in PTX ISA version 7.0.

Support for `.level::cache_hint` and `.level::prefetch_size` qualifiers introduced in PTX ISA
version 7.4.

Support for `ignore-src` operand introduced in PTX ISA version 7.5.

Support for sub-qualifier `::cta` introduced in PTX ISA version 7.8.

Target ISA Notes

Requires `sm_80` or higher.

Sub-qualifier `::cta` requires `sm_30` or higher.

Examples

```
cp.async.ca.shared.global  [shrd],    [gbl + 4], 4;

cp.async.ca.shared::cta.global  [%r0 + 8], [%r1],     8;

cp.async.cg.shared.global  [%r2],     [%r3],     16;



cp.async.cg.shared.global.L2::64B   [%r2],      [%r3],     16;

cp.async.cg.shared.global.L2::128B  [%r0 + 16], [%r1],     16;

cp.async.cg.shared.global.L2::256B  [%r2 + 32], [%r3],     16;



createpolicy.fractional.L2::evict_last.L2::evict_unchanged.b64 cache-policy, 0.25;

cp.async.ca.shared.global.L2::cache_hint [%r2], [%r1], 4, cache-policy;



cp.async.ca.shared.global                   [shrd], [gbl], 4, p;

cp.async.cg.shared.global.L2::cache_hint   [%r0], [%r2], 16, q, cache-policy;
```

###### 9.7.9.25.3.2. [Data Movement and Conversion Instructions: `cp.async.commit_group`](#data-movement-and-conversion-instructions-cp-async-commit-group)[](#data-movement-and-conversion-instructions-cp-async-commit-group "Permalink to this headline")

`cp.async.commit_group`

Commits all prior initiated but uncommitted `cp.async` instructions into a *cp.async-group*.

Syntax

```
cp.async.commit_group ;
```

Description

`cp.async.commit_group` instruction creates a new *cp.async-group* per thread and batches all
prior `cp.async` instructions initiated by the executing thread but not committed to any
*cp.async-group* into the new *cp.async-group*. If there are no uncommitted `cp.async`
instructions then `cp.async.commit_group` results in an empty *cp.async-group.*

An executing thread can wait for the completion of all `cp.async` operations in a *cp.async-group*
using `cp.async.wait_group`.

There is no memory ordering guarantee provided between any two `cp.async` operations within the
same *cp.async-group*. So two or more `cp.async` operations within a *cp.async-group* copying data
to the same location results in undefined behavior.

PTX ISA Notes

Introduced in PTX ISA version 7.0.

Target ISA Notes

Requires `sm_80` or higher.

Examples

```
// Example 1:

cp.async.ca.shared.global [shrd], [gbl], 4;

cp.async.commit_group ; // Marks the end of a cp.async group



// Example 2:

cp.async.ca.shared.global [shrd1],   [gbl1],   8;

cp.async.ca.shared.global [shrd1+8], [gbl1+8], 8;

cp.async.commit_group ; // Marks the end of cp.async group 1



cp.async.ca.shared.global [shrd2],    [gbl2],    16;

cp.async.cg.shared.global [shrd2+16], [gbl2+16], 16;

cp.async.commit_group ; // Marks the end of cp.async group 2
```

###### 9.7.9.25.3.3. [Data Movement and Conversion Instructions: `cp.async.wait_group` / `cp.async.wait_all`](#data-movement-and-conversion-instructions-cp-async-wait-group)[](#data-movement-and-conversion-instructions-cp-async-wait-group "Permalink to this headline")

`cp.async.wait_group`, `cp.async.wait_all`

Wait for completion of prior asynchronous copy operations.

Syntax

```
cp.async.wait_group N;

cp.async.wait_all ;
```

Description

`cp.async.wait_group` instruction will cause executing thread to wait till only `N` or fewer of
the most recent *cp.async-group*s are pending and all the prior *cp.async-group*s committed by
the executing threads are complete. For example, when `N` is 0, the executing thread waits on all
the prior *cp.async-group*s to complete. Operand `N` is an integer constant.

`cp.async.wait_all` is equivalent to :

```
cp.async.commit_group;

cp.async.wait_group 0;
```

An empty *cp.async-group* is considered to be trivially complete.

Writes performed by `cp.async` operations are made visible to the executing thread only after:

1. The completion of `cp.async.wait_all` or
2. The completion of `cp.async.wait_group` on the *cp.async-group* in which the `cp.async`
   belongs to or
3. [mbarrier.test\_wait](#parallel-synchronization-and-communication-instructions-mbarrier-test-wait-try-wait)
   returns `True` on an *mbarrier object* which is tracking the completion of the `cp.async`
   operation.

There is no ordering between two `cp.async` operations that are not synchronized with
`cp.async.wait_all` or `cp.async.wait_group` or [mbarrier objects](#parallel-synchronization-and-communication-instructions-mbarrier).

`cp.async.wait_group` and `cp.async.wait_all` does not provide any ordering and visibility
guarantees for any other memory operation apart from `cp.async`.

PTX ISA Notes

Introduced in PTX ISA version 7.0.

Target ISA Notes

Requires `sm_80` or higher.

Examples

```
// Example of .wait_all:

cp.async.ca.shared.global [shrd1], [gbl1], 4;

cp.async.cg.shared.global [shrd2], [gbl2], 16;

cp.async.wait_all;  // waits for all prior cp.async to complete



// Example of .wait_group :

cp.async.ca.shared.global [shrd3], [gbl3], 8;

cp.async.commit_group;  // End of group 1



cp.async.cg.shared.global [shrd4], [gbl4], 16;

cp.async.commit_group;  // End of group 2



cp.async.cg.shared.global [shrd5], [gbl5], 16;

cp.async.commit_group;  // End of group 3



cp.async.wait_group 1;  // waits for group 1 and group 2 to complete
```