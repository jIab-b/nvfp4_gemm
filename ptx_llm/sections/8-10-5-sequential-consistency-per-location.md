### 8.10.5. Sequential Consistency Per Location 

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