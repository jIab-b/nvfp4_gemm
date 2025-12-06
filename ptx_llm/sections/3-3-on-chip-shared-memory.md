## 3.3. On-chip Shared Memory 

As illustrated by [Figure 4](#set-of-simt-multiprocessors-hardware-model), each multiprocessor has
on-chip memory of the four following types:

* One set of local 32-bit *registers* per processor,
* A parallel data cache or *shared memory* that is shared by all scalar processor cores and is where
  the shared memory space resides,
* A read-only *constant cache* that is shared by all scalar processor cores and speeds up reads from
  the constant memory space, which is a read-only region of device memory,
* A read-only *texture cache* that is shared by all scalar processor cores and speeds up reads from
  the texture memory space, which is a read-only region of device memory; each multiprocessor
  accesses the texture cache via a *texture unit* that implements the various addressing modes and
  data filtering.

The local and global memory spaces are read-write regions of device memory.