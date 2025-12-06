##### 9.7.16.10.2. Matrix Layout Organization 

[Table 51](#tcgen05-matrices-majorness) describes the major-ness used for different matrices.

Table 51 Major-ness for different matrices[](#tcgen05-matrices-majorness "Permalink to this table")





| Matrix | Residing in Memory | Default Major-ness |
| --- | --- | --- |
| D | Tensor Memory | Row-Major |
| A | Tensor Memory |
| Shared Memory | Depends on swizzling mode. Refer [Shared Memory Layout and Swizzling](#tcgen05-shared-memory-layout-swizzling) |
| B | Shared Memory |