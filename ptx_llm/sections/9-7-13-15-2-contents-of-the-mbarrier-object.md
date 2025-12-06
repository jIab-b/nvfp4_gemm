##### 9.7.13.15.2. Contents of the mbarrier object 

An opaque *mbarrier object* keeps track of the following information :

* Current phase of the *mbarrier object*
* Count of pending arrivals for the current phase of the *mbarrier object*
* Count of expected arrivals for the next phase of the *mbarrier object*
* Count of pending asynchronous memory operations (or transactions) tracked by the current phase of
  the *mbarrier object*. This is also referred to as *tx-count*.

An *mbarrier object* progresses through a sequence of phases where each phase is defined by threads
performing an expected number of
[arrive-on](#parallel-synchronization-and-communication-instructions-mbarrier-arrive-on)
operations.

The valid range of each of the counts is as shown below:

| Count name | Minimum value | Maximum value |
| --- | --- | --- |
| Expected arrival count | 1 | 220 - 1 |
| Pending arrival count | 0 | 220 - 1 |
| tx-count | -(220 - 1) | 220 - 1 |