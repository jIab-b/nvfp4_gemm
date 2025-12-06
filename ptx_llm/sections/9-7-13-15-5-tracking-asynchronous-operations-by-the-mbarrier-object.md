##### 9.7.13.15.5. Tracking asynchronous operations by the mbarrier object 

Starting with the Hopper architecture (`sm_9x`), *mbarrier object* supports a new count, called
*tx-count*, which is used for tracking the completion of asynchronous memory operations or
transactions. *tx-count* tracks the number of asynchronous transactions, in units specified by the
asynchronous memory operation, that are outstanding and yet to be complete.

The *tx-count* of an *mbarrier object* must be set to the total amount of asynchronous memory
operations, in units as specified by the asynchronous operations, to be tracked by the current
phase. Upon completion of each of the asynchronous operations, the [complete-tx](#parallel-synchronization-and-communication-instructions-mbarrier-complete-tx-operation)
operation will be performed on the *mbarrier object* and thus progress the mbarrier towards the
completion of the current phase.

###### 9.7.13.15.5.1. [expect-tx operation](#parallel-synchronization-and-communication-instructions-mbarrier-expect-tx-operation)[](#parallel-synchronization-and-communication-instructions-mbarrier-expect-tx-operation "Permalink to this headline")

The *expect-tx* operation, with an `expectCount` argument, increases the *tx-count* of an
*mbarrier object* by the value specified by `expectCount`. This sets the current phase of the
*mbarrier object* to expect and track the completion of additional asynchronous transactions.

###### 9.7.13.15.5.2. [complete-tx operation](#parallel-synchronization-and-communication-instructions-mbarrier-complete-tx-operation)[](#parallel-synchronization-and-communication-instructions-mbarrier-complete-tx-operation "Permalink to this headline")

The *complete-tx* operation, with an `completeCount` argument, on an *mbarrier object* consists of the following:

mbarrier signaling
:   Signals the completion of asynchronous transactions that were tracked by the current phase. As a
    result of this, *tx-count* is decremented by `completeCount`.

mbarrier potentially completing the current phase
:   If the current phase has been completed then the mbarrier transitions to the next phase. Refer to
    [Phase Completion of the mbarrier object](#parallel-synchronization-and-communication-instructions-mbarrier-phase-completion)
    for details on phase completion requirements and phase transition process.