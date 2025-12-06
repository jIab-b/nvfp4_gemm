##### 9.7.16.5.1. CTA Pair 

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