##### 9.7.16.6.2. Pipelined tcgen05 Instructions 

The asynchronous `tcgen05` operations may execute and complete in a different order than they
were issued. However, some specific pairs of the asynchronous `tcgen05` instructions form
`tcgen05` pipelines, where in the two asynchronous operations are guaranteed to execute in
the same order as the instructions that issued them. The specific pairings are as follows:

1. `tcgen05.mma.cta_group::N` -> `tcgen05.mma.cta_group::N` (same N and accumulator and shape)
2. `tcgen05.copy.cta_group::N` -> `tcgen05.mma.cta_group::N` (same N)
3. `tcgen05.shift.cta_group::N` -> `tcgen05.mma.cta_group::N` (same N)
4. `tcgen05.shift.cta_group::N` -> `tcgen05.cp.4x256b.cta_group::N` (same N)
5. `tcgen05.mma.cta_group::N` -> `tcgen05.shift.cta_group::N` (same N)

###### 9.7.16.6.2.1. [Implicitly pipelined tcgen05 Instructions](#tcgen05-memory-consistency-model-pipelined-instructions-implicit)[](#tcgen05-memory-consistency-model-pipelined-instructions-implicit "Permalink to this headline")

Instructions `tcgen05.commit` and `tcgen05.wait` are implicitly pipelined with respect
to previously issued `tcgen05.{mma,cp,shift}` and `tcgen05.{ld,st}` instructions
respectively that they track from the same thread.

###### 9.7.16.6.2.1.1. [mbarrier based completion mechanism](#tcgen05-memory-consistency-model-mbarrier-completion)[](#tcgen05-memory-consistency-model-mbarrier-completion "Permalink to this headline")

Completion of the following instruction’s asynchronous operations is observed
through the mbarrier based waiting mechanism:

1. `tcgen05.mma`
2. `tcgen05.cp`
3. `tcgen05.shift`

`tcgen05.commit` is used to track the completion of the above asynchronous instructions.

Following are the implicitly pipelined `tcgen05` instruction pairing that uses mbarrier
based completion mechanism:

* `tcgen05.mma.cta_group::N` -> `tcgen05.commit.cta_group::N` (same N)
* `tcgen05.cp.cta_group::N` -> `tcgen05.commit.cta_group::N` (same N)
* `tcgen05.shift.cta_group::N` -> `tcgen05.commit.cta_group::N` (same N)

###### 9.7.16.6.2.1.2. [`tcgen05.wait` instruction based completion mechanism](#tcgen05-memory-consistency-model-wait-completion)[](#tcgen05-memory-consistency-model-wait-completion "Permalink to this headline")

Completion of the following instruction’s asynchronous operations is observed through
`tcgen05.wait` based waiting mechanism:

1. `tcgen05.ld`
2. `tcgen05.st`

`tcgen05.wait::ld` and `tcgen05.wait::st` is used to track the completion of the
`tcgen05.ld` and `tcgen05.st` asynchronous instructions.

Following are the implicitly pipelined `tcgen05` instruction pairing that uses
`tcgen05.wait` based completion mechanism:

* `tcgen05.ld` -> `tcgen05.wait::ld`
* `tcgen05.st` -> `tcgen05.wait::st`