##### 9.7.13.15.3. Lifecycle of the mbarrier object 

The *mbarrier object* must be initialized prior to use.

An *mbarrier object* is used to synchronize threads and asynchronous memory operations.

An *mbarrier object* may be used to perform a sequence of such synchronizations.

An *mbarrier object* must be invalidated to repurpose its memory for any purpose,
including repurposing it for another mbarrier object.