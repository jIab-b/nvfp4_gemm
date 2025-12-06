### 8.10.3. Atomicity 

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