###### 9.7.16.6.4.4. Non-pipelined instructions, different thread 

In this pattern, the producer threads that issue the asynchronous `tcgen05` instructions
must explicitly wait for the instructions’ completion before synchronizing with the consumer threads.

Example 1:

| Thread 0 | Thread 1 |
| --- | --- |
| ``` tcgen05.ld  tcgen05.wait::ld  tcgen05.fence::before_thread_sync  mbarrier.arrive.relaxed.cluster ``` |  |
|  | ``` mbarrier.try_wait.relaxed.cluster // loop till success  tcgen05.fence::after_thread_sync  tcgen05.mma ``` |

Example 1:

| Thread 0 | Thread 1 |
| --- | --- |
| ``` tcgen05.mma  tcgen05.commit.mbarrier::arrive::one [mbar] ``` |  |
|  | ``` mbarrier.try_wait.relaxed.cluster [mbar] // loop till success  tcgen05.fence::after_thread_sync  tcgen05.ld ``` |

The synchronization mechanisms can also be composed with each other. For example:

| Thread 0 | Thread 1 |
| --- | --- |
| ``` tcgen05.mma  tcgen05.commit.mbarrier::arrive::one [bar1]  mbarrier.try_wait.relaxed.cluster [bar1] // loop  ...  tcgen05.fence::after_thread_sync  ...// completion is guaranteed  tcgen05.fence::before_thread_sync  mbarrier.arrive.relaxed.cluster [bar2] // loop  ... ``` |  |
|  | ``` mbarrier.try_wait.relaxed.cluster [bar2] // loop  ...  tcgen05.fence::after_thread_sync  tcgen05.ld ``` |