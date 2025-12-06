###### 9.7.9.25.1.2. Mbarrier-based mechanism 

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