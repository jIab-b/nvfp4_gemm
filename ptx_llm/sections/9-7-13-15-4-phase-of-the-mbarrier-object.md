##### 9.7.13.15.4. Phase of the mbarrier object 

The phase of an *mbarrier object* is the number of times the *mbarrier object* has been used to
synchronize threads and [asynchronous](#program-order-async-operations)
operations. In each phase {0, 1, 2, …}, threads perform in program order :

* [arrive-on](#parallel-synchronization-and-communication-instructions-mbarrier-arrive-on)
  operations to complete the current phase and
* *test\_wait* / *try\_wait* operations to check for the completion of the current phase.

An *mbarrier object* is automatically reinitialized upon completion of the current phase for
immediate use in the next phase. The current phase is incomplete and all prior phases are complete.

For each phase of the mbarrier object, at least one *test\_wait* or *try\_wait* operation must be
performed which returns `True` for `waitComplete` before an [arrive-on](#parallel-synchronization-and-communication-instructions-mbarrier-arrive-on) operation
in the subsequent phase.