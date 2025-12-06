##### 9.7.9.25.1. Completion Mechanisms for Asynchronous Copy Operations 

A thread must explicitly wait for the completion of an asynchronous copy operation in order to
access the result of the operation. Once an asynchronous copy operation is initiated, modifying the
source memory location or tensor descriptor or reading from the destination memory location before
the asynchronous operation completes, exhibits undefined behavior.

This section describes two asynchronous copy operation completion mechanisms supported in PTX:
Async-group mechanism and mbarrier-based mechanism.

Asynchronous operations may be tracked by either of the completion mechanisms or both mechanisms.
The tracking mechanism is instruction/instruction-variant specific.

###### 9.7.9.25.1.1. [Async-group mechanism](#data-movement-and-conversion-instructions-asynchronous-copy-completion-mechanisms-async-group)[](#data-movement-and-conversion-instructions-asynchronous-copy-completion-mechanisms-async-group "Permalink to this headline")

When using the async-group completion mechanism, the issuing thread specifies a group of
asynchronous operations, called *async-group*, using a *commit* operation and tracks the completion
of this group using a *wait* operation. The thread issuing the asynchronous operation must create
separate *async-groups* for bulk and non-bulk asynchronous operations.

A *commit* operation creates a per-thread *async-group* containing all prior asynchronous operations
tracked by *async-group* completion and initiated by the executing thread but none of the asynchronous
operations following the commit operation. A committed asynchronous operation belongs to a single
*async-group*.

When an *async-group* completes, all the asynchronous operations belonging to that group are
complete and the executing thread that initiated the asynchronous operations can read the result of
the asynchronous operations. All *async-groups* committed by an executing thread always complete in
the order in which they were committed. There is no ordering between asynchronous operations within
an *async-group*.

A typical pattern of using *async-group* as the completion mechanism is as follows:

* Initiate the asynchronous operations.
* Group the asynchronous operations into an *async-group* using a *commit* operation.
* Wait for the completion of the async-group using the wait operation.
* Once the *async-group* completes, access the results of all asynchronous operations in that
  *async-group*.

###### 9.7.9.25.1.2. [Mbarrier-based mechanism](#data-movement-and-conversion-instructions-asynchronous-copy-completion-mechanisms-mbarrier)[](#data-movement-and-conversion-instructions-asynchronous-copy-completion-mechanisms-mbarrier "Permalink to this headline")

A thread can track the completion of one or more asynchronous operations using the current phase of
an *mbarrier object*. When the current phase of the *mbarrier object* is complete, it implies that
all asynchronous operations tracked by this phase are complete, and all threads participating in
that *mbarrier object* can access the result of the asynchronous operations.

The *mbarrier object* to be used for tracking the completion of an asynchronous operation can be
either specified along with the asynchronous operation as part of its syntax, or as a separate
operation. For a bulk asynchronous operation, the *mbarrier object* must be specified in the
asynchronous operation, whereas for non-bulk operations, it can be specified after the asynchronous
operation.

A typical pattern of using mbarrier-based completion mechanism is as follows:

* Initiate the asynchronous operations.
* Set up an *mbarrier object* to track the asynchronous operations in its current phase, either as
  part of the asynchronous operation or as a separate operation.
* Wait for the *mbarrier object* to complete its current phase using `mbarrier.test_wait` or
  `mbarrier.try_wait`.
* Once the `mbarrier.test_wait` or `mbarrier.try_wait` operation returns `True`, access the
  results of the asynchronous operations tracked by the *mbarrier object*.