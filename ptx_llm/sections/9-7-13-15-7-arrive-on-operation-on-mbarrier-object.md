##### 9.7.13.15.7. Arrive-on operation on mbarrier object 

An *arrive-on* operation, with an optional *count* argument, on an *mbarrier object* consists of the
following 2 steps :

* mbarrier signalling:

  Signals the arrival of the executing thread OR completion of the asynchronous instruction which
  signals the arrive-on operation initiated by the executing thread on the *mbarrier object*. As a
  result of this, the pending arrival count is decremented by *count*. If the *count* argument is
  not specified, then it defaults to 1.
* mbarrier potentially completing the current phase:

  If the current phase has been completed then the mbarrier transitions to the next phase. Refer to
  [Phase Completion of the mbarrier object](#parallel-synchronization-and-communication-instructions-mbarrier-phase-completion)
  for details on phase completion requirements and phase transition process.