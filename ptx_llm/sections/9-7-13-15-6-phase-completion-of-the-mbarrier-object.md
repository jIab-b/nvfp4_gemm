##### 9.7.13.15.6. Phase Completion of the mbarrier object 

The requirements for completion of the current phase are described below. Upon completion of the
current phase, the phase transitions to the subsequent phase as described below.

Current phase completion requirements
:   An *mbarrier object* completes the current phase when all of the following conditions are met:

    * The count of the pending arrivals has reached zero.
    * The *tx-count* has reached zero.

Phase transition
:   When an *mbarrier* object completes the current phase, the following actions are performed
    atomically:

    * The *mbarrier object* transitions to the next phase.
    * The pending arrival count is reinitialized to the expected arrival count.