## 8.10. Axioms 

### 8.10.1. [Coherence](#coherence-axiom)[](#coherence-axiom "Permalink to this headline")

If a write W precedes an *overlapping* write W’ in *causality order*, then W must precede W’ in
*coherence order*.

### 8.10.2. [Fence-SC](#fence-sc-axiom)[](#fence-sc-axiom "Permalink to this headline")

*Fence-SC* order cannot contradict *causality order*. For a pair of *morally strong* *fence.sc*
operations F1 and F2, if F1 precedes F2 in *causality order*, then F1 must precede F2 in *Fence-SC*
order.

### 8.10.3. [Atomicity](#atomicity-axiom)[](#atomicity-axiom "Permalink to this headline")

Single-Copy Atomicity

Conflicting *morally strong* operations are performed with *single-copy atomicity*. When a read R
and a write W are *morally strong*, then the following two communications cannot both exist in the
same execution, for the set of bytes accessed by both R and W:

1. R reads any byte from W.
2. R reads any byte from any write W’ which precedes W in *coherence order*.

Atomicity of read-modify-write (RMW) operations

When an *atomic* operation A and a write W *overlap* and are *morally strong*, then the following
two communications cannot both exist in the same execution, for the set of bytes accessed by both A
and W:

1. A reads any byte from a write W’ that precedes W in *coherence order*.
2. A follows W in *coherence order*.

Litmus Test 1

|  |  |
| --- | --- |
| ``` .global .u32 x = 0; ``` | |
| T1 | T2 |
| ``` A1: atom.sys.inc.u32 %r0, [x]; ``` | ``` A2: atom.sys.inc.u32 %r0, [x]; ``` |
| ``` FINAL STATE: x == 2 ``` | |

Atomicity is guaranteed when the operations are *morally strong*.

Litmus Test 2

|  |  |
| --- | --- |
| ``` .global .u32 x = 0; ``` | |
| T1 | T2 (In a different CTA) |
| ``` A1: atom.cta.inc.u32 %r0, [x]; ``` | ``` A2: atom.gpu.inc.u32 %r0, [x]; ``` |
| ``` FINAL STATE: x == 1 OR x == 2 ``` | |

Atomicity is not guaranteed if the operations are not *morally strong*.

### 8.10.4. [No Thin Air](#no-thin-air-axiom)[](#no-thin-air-axiom "Permalink to this headline")

Values may not appear “out of thin air”: an execution cannot speculatively produce a value in such a
way that the speculation becomes self-satisfying through chains of instruction dependencies and
inter-thread communication. This matches both programmer intuition and hardware reality, but is
necessary to state explicitly when performing formal analysis.

Litmus Test: Load Buffering

|  |  |
| --- | --- |
| ``` .global .u32 x = 0;  .global .u32 y = 0; ``` | |
| T1 | T2 |
| ``` A1: ld.global.u32 %r0, [x];  B1: st.global.u32 [y], %r0; ``` | ``` A2: ld.global.u32 %r1, [y];  B2: st.global.u32 [x], %r1; ``` |
| ``` FINAL STATE: x == 0 AND y == 0 ``` | |

The litmus test known as “LB” (Load Buffering) checks such forbidden values that may arise out of
thin air. Two threads T1 and T2 each read from a first variable and copy the observed result into a
second variable, with the first and second variable exchanged between the threads. If each variable
is initially zero, the final result shall also be zero. If A1 reads from B2 and A2 reads from B1,
then values passing through the memory operations in this example form a cycle:
A1->B1->A2->B2->A1. Only the values x == 0 and y == 0 are allowed to satisfy this cycle. If any of
the memory operations in this example were to speculatively associate a different value with the
corresponding memory location, then such a speculation would become self-fulfilling, and hence
forbidden.

### 8.10.5. [Sequential Consistency Per Location](#sc-per-loc-axiom)[](#sc-per-loc-axiom "Permalink to this headline")

Within any set of *overlapping* memory operations that are pairwise *morally strong*, *communication
order* cannot contradict *program order*, i.e., a concatenation of *program order* between
*overlapping* operations and *morally strong* relations in *communication order* cannot result in a
cycle. This ensures that each program slice of *overlapping* pairwise morally *strong operations* is
strictly *sequentially-consistent*.

Litmus Test: CoRR

|  |  |
| --- | --- |
| ``` .global .u32 x = 0; ``` | |
| T1 | T2 |
| ``` W1: st.global.relaxed.sys.u32 [x], 1; ``` | ``` R1: ld.global.relaxed.sys.u32 %r0, [x];  R2: ld.global.relaxed.sys.u32 %r1, [x]; ``` |
| ``` IF %r0 == 1 THEN %r1 == 1 ``` | |

The litmus test “CoRR” (Coherent Read-Read), demonstrates one consequence of this guarantee. A
thread T1 executes a write W1 on a location x, and a thread T2 executes two (or an infinite sequence
of) reads R1 and R2 on the same location x. No other writes are executed on x, except the one
modelling the initial value. The operations W1, R1 and R2 are pairwise *morally strong*. If R1 reads
from W1, then the subsequent read R2 must also observe the same value. If R2 observed the initial
value of x instead, then this would form a sequence of *morally-strong* relations R2->W1->R1 in
*communication order* that contradicts the *program order* R1->R2 in thread T2. Hence R2 cannot read
the initial value of x in such an execution.

### 8.10.6. [Causality](#causality-axiom)[](#causality-axiom "Permalink to this headline")

Relations in *communication order* cannot contradict *causality order*. This constrains the set of
candidate write operations that a read operation may read from:

1. If a read R precedes an *overlapping* write W in *causality order*, then R cannot read from W.
2. If a write W precedes an *overlapping* read R in *causality order*, then for any byte accessed by
   both R and W, R cannot read from any write W’ that precedes W in *coherence order*.

Litmus Test: Message Passing

|  |  |
| --- | --- |
| ``` .global .u32 data = 0;  .global .u32 flag = 0; ``` | |
| T1 | T2 |
| ``` W1: st.global.u32 [data], 1;  F1: fence.sys;  W2: st.global.relaxed.sys.u32 [flag], 1; ``` | ``` R1: ld.global.relaxed.sys.u32 %r0, [flag];  F2: fence.sys;  R2: ld.global.u32 %r1, [data]; ``` |
| ``` IF %r0 == 1 THEN %r1 == 1 ``` | |

The litmus test known as “MP” (Message Passing) represents the essence of typical synchronization
algorithms. A vast majority of useful programs can be reduced to sequenced applications of this
pattern.

Thread T1 first writes to a data variable and then to a flag variable while a second thread T2 first
reads from the flag variable and then from the data variable. The operations on the flag are
*morally strong* and the memory operations in each thread are separated by a *fence*, and these
*fences* are *morally strong*.

If R1 observes W2, then the release pattern “F1; W2” *synchronizes* with the *acquire pattern* “R1;
F2”. This establishes the *causality order* W1 -> F1 -> W2 -> R1 -> F2 -> R2. Then axiom *causality*
guarantees that R2 cannot read from any write that precedes W1 in *coherence order*. In the absence
of any other writes in this example, R2 must read from W1.

Litmus Test: CoWR

|  |
| --- |
| ``` // These addresses are aliases  .global .u32 data_alias_1;  .global .u32 data_alias_2; ``` |
| T1 |
| ``` W1: st.global.u32 [data_alias_1], 1;  F1: fence.proxy.alias;  R1: ld.global.u32 %r1, [data_alias_2]; ``` |
| ``` %r1 == 1 ``` |

Virtual aliases require an alias *proxy fence* along the synchronization path.

Litmus Test: Store Buffering

The litmus test known as “SB” (Store Buffering) demonstrates the *sequential consistency* enforced
by the `fence.sc`. A thread T1 writes to a first variable, and then reads the value of a second
variable, while a second thread T2 writes to the second variable and then reads the value of the
first variable. The memory operations in each thread are separated by `fence.`sc instructions,
and these *fences* are *morally strong*.

|  |  |
| --- | --- |
| ``` .global .u32 x = 0;  .global .u32 y = 0; ``` | |
| T1 | T2 |
| ``` W1: st.global.u32 [x], 1;  F1: fence.sc.sys;  R1: ld.global.u32 %r0, [y]; ``` | ``` W2: st.global.u32 [y], 1;  F2: fence.sc.sys;  R2: ld.global.u32 %r1, [x]; ``` |
| ``` %r0 == 1 OR %r1 == 1 ``` | |

In any execution, either F1 precedes F2 in *Fence-SC* order, or vice versa. If F1 precedes F2 in
*Fence-SC* order, then F1 *synchronizes* with F2. This establishes the *causality order* in W1 -> F1
-> F2 -> R2. Axiom *causality* ensures that R2 cannot read from any write that precedes W1 in
*coherence order*. In the absence of any other write to that variable, R2 must read from
W1. Similarly, in the case where F2 precedes F1 in *Fence-SC* order, R1 must read from W2. If each
`fence.sc` in this example were replaced by a `fence.acq_rel` instruction, then this outcome is
not guaranteed. There may be an execution where the write from each thread remains unobserved from
the other thread, i.e., an execution is possible, where both R1 and R2 return the initial value “0”
for variables y and x respectively.