### 8.4.2. volatile Operation 

A *volatile* operation is a memory operation with `.volatile` qualifier specified.
The semantics of volatile operations are equivalent to a relaxed memory operation with system-scope
but with the following extra implementation-specific constraints:

* The number of volatile *instructions* (not operations) executed by a program is preserved.
  Hardware may combine and merge volatile *operations* issued by multiple different volatile
  *instructions*, that is, the number of volatile *operations* in the program is not preserved.
* Volatile *instructions* are not re-ordered around other volatile *instructions*, but the memory
  *operations* performed by those *instructions* may be re-ordered around each other.

Note

PTX volatile operations are intended for compilers to lower volatile read and write operations from
CUDA C++, and other programming languages sharing CUDA C++ volatile semantics, to PTX.

Since volatile operations are relaxed at system-scope with extra constraints, prefer using other
*strong* read or write operations (e.g. `ld.relaxed.sys` or `st.relaxed.sys`) for
**Inter-Thread Synchronization** instead, which may deliver better performance.

PTX volatile operations are not suited for **Memory Mapped IO (MMIO)** because volatile operations
do not preserve the number of memory operations performed, and may perform more or less operations
than requested in a non-deterministic way.
Use [.mmio operations](#mmio-operation) instead, which strictly preserve the number of operations
performed.