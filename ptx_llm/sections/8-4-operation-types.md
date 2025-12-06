## 8.4. Operation types 

For simplicity, the rest of the document refers to the following operation types, instead of
mentioning specific instructions that give rise to them.

Table 20 Operation Types[](#id673 "Permalink to this table")




| Operation Type | Instruction/Operation |
| --- | --- |
| atomic operation | `atom` or `red` instruction. |
| read operation | All variants of `ld` instruction and `atom` instruction (but not `red` instruction). |
| write operation | All variants of `st` instruction, and *atomic* operations if they result in a write. |
| memory operation | A *read* or *write* operation. |
| volatile operation | An instruction with `.volatile` qualifier. |
| acquire operation | A *memory* operation with `.acquire` or `.acq_rel` qualifier. |
| release operation | A *memory* operation with `.release` or `.acq_rel` qualifier. |
| mmio operation | An `ld` or `st` instruction with `.mmio` qualifier. |
| memory fence operation | A `membar`, `fence.sc` or `fence.acq_rel` instruction. |
| proxy fence operation | A `fence.proxy` or a `membar.proxy` instruction. |
| strong operation | A *memory fence* operation, or a *memory* operation with a `.relaxed`, `.acquire`, `.release`, `.acq_rel`, `.volatile`, or `.mmio` qualifier. |
| weak operation | An `ld` or `st` instruction with a `.weak` qualifier. |
| synchronizing operation | A `barrier` instruction, *fence* operation, *release* operation or *acquire* operation. |

### 8.4.1. [mmio Operation](#mmio-operation)[](#mmio-operation "Permalink to this headline")

An *mmio* operation is a memory operation with `.mmio` qualifier specified. It is usually performed
on a memory location which is mapped to the control registers of peer I/O devices. It can also be
used for communication between threads but has poor performance relative to non-*mmio* operations.

The semantic meaning of *mmio* operations cannot be defined precisely as it is defined by the
underlying I/O device. For formal specification of semantics of *mmio* operation from Memory
Consistency Model perspective, it is equivalent to the semantics of a *strong* operation. But it
follows a few implementation-specific properties, if it meets the *CUDA atomicity requirements* at
the specified scope:

* Writes are always performed and are never combined within the scope specified.
* Reads are always performed, and are not forwarded, prefetched, combined, or allowed to hit any
  cache within the scope specified.

  + As an exception, in some implementations, the surrounding locations may also be loaded. In such
    cases the amount of data loaded is implementation specific and varies between 32 and 128 bytes
    in size.

### 8.4.2. [volatile Operation](#volatile-operation)[](#volatile-operation "Permalink to this headline")

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