##### 9.7.16.1.2. Tensor Memory Allocation 

The Tensor Memory is dynamically allocated. The Tensor Memory must be allocated by a single
warp in a CTA using the
[Tensor Memory Allocation and Management Instructions](#tcgen05-memory-alloc-manage-instructions).

The allocation and deallocation of [Tensor Memory](#tensor-memory) is performed in terms of
columns. The unit of allocation is 32 columns and the number of columns being allocated must be
a power of 2. When a column is allocated, all 128 lanes of the column are allocated.

All of the Tensor Memory that was allocated in a kernel, must be explicitly deallocated
before the kernel exits.