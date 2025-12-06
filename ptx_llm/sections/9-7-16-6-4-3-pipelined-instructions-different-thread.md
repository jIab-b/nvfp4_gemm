###### 9.7.16.6.4.3. Pipelined instructions, different thread 

In this pattern, no explicit waiting mechanism is needed but proper synchronization between threads is needed.

Example:

| Thread 0 | Thread 1 |
| --- | --- |
| ``` tcgen05.cp  tcgen05.fence::before_thread_sync  mbarrier.arrive.relaxed.cluster ``` |  |
|  | ``` mbarrier.try_wait.relaxed.cluster // loop till success  tcgen05.fence::after_thread_sync  tcgen05.mma ``` |