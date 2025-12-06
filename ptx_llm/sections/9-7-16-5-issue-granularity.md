#### 9.7.16.5. Issue Granularity 

Each of the `tcgen05` operation has different requirements for the number of
threads/warps that needs to issue them.

The following table lists the execution granularity requirements of each of the
`tcgen05` operation:

Table 46 Execution granularity requirements for tcgen05 operations[](#tcgen05-ops-execution-granularity "Permalink to this table")





| tcgen05 operation | .cta\_group | Issue Granularity |
| --- | --- | --- |
| ``` .mma,  .cp,  .shift,  .commit ``` | ::1 | An issue from a single thread in the current CTA would initiate the base operation. |
| ::2 | Issue from a single thread from the [CTA-Pair](#tcgen05-cta-pair) would initiate the base operation. When the current CTA issues the operation, the peer CTA should be active and should not have exited. |
| ``` .alloc,  .dealloc,  .relinquish_alloc_permit ``` | ::1 | Issue from a single warp in the current CTA would initiate the allocation management instruction. |
| ::2 | Issue from two warps, one in each of the current CTA and its [Peer CTA](#tcgen05-peer-cta), collectively needs to perform the operation. When the current CTA issues the operation, the peer CTA should be active and should not have exited. |
| ``` .ld,  .st,  .wait::{ld, st} ``` | N/A | Issue from a warp in the current CTA can access only 1/4 of the Tensor Memory of the current CTA. So, a warpgroup is needed to access the entire Tensor Memory of the current CTA. |
| ``` .fence::* ``` | N/A | A thread needs to fence all its accesses to the tensor memory that it wants to order with other accesses to the tensor memory from other threads. |

##### 9.7.16.5.1. [CTA Pair](#tcgen05-cta-pair)[](#tcgen05-cta-pair "Permalink to this headline")

Any 2 CTAs within the cluster whose `%cluster_ctarank` differs by the last bit only
is said to form a CTA pair.

Within a CTA pair, the CTA whose last bit in the `%cluster_ctarank` is:

* 0 is termed the even numbered CTA within the CTA pair.
* 1 is termed as the odd numbered CTA within the CTA pair.

Most of the `tcgen05` operations can either execute at a single CTA level granularity OR
at a CTA pair level granularity. When a `tcgen05` operation is performed at CTA pair
granularity, the Tensor Memory of both the CTAs within the CTA pair are accessed. The set
of threads that need to issue the `tcgen05` operation is listed in the
[Issue Granularity](#tcgen05-issue-granularity).

##### 9.7.16.5.2. [Peer CTA](#tcgen05-peer-cta)[](#tcgen05-peer-cta "Permalink to this headline")

The peer CTA of the odd CTA within the CTA pair is the even CTA in the same pair.
Similarly, the peer CTA of the even CTA within the CTA pair is the odd CTA in the same pair.