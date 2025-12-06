###### 9.7.13.15.5.1. expect-tx operation 

The *expect-tx* operation, with an `expectCount` argument, increases the *tx-count* of an
*mbarrier object* by the value specified by `expectCount`. This sets the current phase of the
*mbarrier object* to expect and track the completion of additional asynchronous transactions.