###### 9.7.9.25.1.1. Async-group mechanism 

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