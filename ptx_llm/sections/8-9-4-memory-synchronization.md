### 8.9.4. Memory synchronization 

Synchronizing operations performed by different threads synchronize with each other at runtime as
described here. The effect of such synchronization is to establish *causality order* across threads.

1. A `fence.sc` operation X *synchronizes* with a `fence.sc` operation Y if X precedes Y in the
   *Fence-SC* order.
2. A `bar{.cta}.sync` or `bar{.cta}.red` or `bar{.cta}.arrive` operation *synchronizes* with a
   `bar{.cta}.sync` or `bar{.cta}.red` operation executed on the same barrier.
3. A `barrier.cluster.arrive` operation synchronizes with a `barrier.cluster.wait` operation.
4. A *release* pattern X *synchronizes* with an *acquire* pattern Y, if a *write* operation in X
   precedes a *read* operation in Y in *observation order*, and the first operation in X and the
   last operation in Y are *morally strong*.

API synchronization

A *synchronizes* relation can also be established by certain CUDA APIs.

1. Completion of a task enqueued in a CUDA stream *synchronizes* with the start of the following
   task in the same stream, if any.
2. For purposes of the above, recording or waiting on a CUDA event in a stream, or causing a
   cross-stream barrier to be inserted due to `cudaStreamLegacy`, enqueues tasks in the associated
   streams even if there are no direct side effects. An event record task *synchronizes* with
   matching event wait tasks, and a barrier arrival task *synchronizes* with matching barrier wait
   tasks.
3. Start of a CUDA kernel *synchronizes* with start of all threads in the kernel. End of all threads
   in a kernel *synchronize* with end of the kernel.
4. Start of a CUDA graph *synchronizes* with start of all source nodes in the graph. Completion of
   all sink nodes in a CUDA graph *synchronizes* with completion of the graph. Completion of a graph
   node *synchronizes* with start of all nodes with a direct dependency.
5. Start of a CUDA API call to enqueue a task *synchronizes* with start of the task.
6. Completion of the last task queued to a stream, if any, *synchronizes* with return from
   `cudaStreamSynchronize`. Completion of the most recently queued matching event record task, if
   any, *synchronizes* with return from `cudaEventSynchronize`. Synchronizing a CUDA device or
   context behaves as if synchronizing all streams in the context, including ones that have been
   destroyed.
7. Returning `cudaSuccess` from an API to query a CUDA handle, such as a stream or event, behaves
   the same as return from the matching synchronization API.

In addition to establishing a *synchronizes* relation, the CUDA API synchronization mechanisms above
also participate in *proxy-preserved base causality order*.