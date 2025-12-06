##### 9.7.16.6.3. Specialized Inter-thread Synchronization for tcgen05 instructions 

The `tcgen05` instructions support a specialized inter-thread synchronization which are
optimized for `tcgen05` family of instructions. The standard memory consistency model
synchronization mechanisms also apply to the `tcgen05` family of instructions.

The [TensorCore 5th Generation Specialized Synchronization Operations](#tcgen05-special-sync-operations) section contains the specialized inter-thread
synchronization for tcgen05 instructions.

The `tcgen05.fence::before_thread_sync` and `tcgen05.fence::after_thread_sync` composes
with execution ordering instructions, like morally strong `ld`/`st`/`atom` instructions,
`mbarrier` instruction, `barrier` instructions and so on, to establish an ordering between
the `tcgen05` operations across threads. The asynchronous `tcgen05` instructions that are
ordered across threads also form a `tcgen05` pipeline.

An asynchronous `tcgen05` operation prior to a `tcgen05.fence::before_thread_sync` is ordered
before all subsequent `tcgen05` and the execution ordering operations.

An asynchronous `tcgen05` operation subsequent to a `tcgen05.fence::after_thread_sync` is
ordered after all the prior `tcgen05` and the execution ordering operations.