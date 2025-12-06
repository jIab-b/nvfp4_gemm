###### 9.7.13.15.5.2. complete-tx operation 

The *complete-tx* operation, with an `completeCount` argument, on an *mbarrier object* consists of the following:

mbarrier signaling
:   Signals the completion of asynchronous transactions that were tracked by the current phase. As a
    result of this, *tx-count* is decremented by `completeCount`.

mbarrier potentially completing the current phase
:   If the current phase has been completed then the mbarrier transitions to the next phase. Refer to
    [Phase Completion of the mbarrier object](#parallel-synchronization-and-communication-instructions-mbarrier-phase-completion)
    for details on phase completion requirements and phase transition process.