##### 9.7.16.6.5. Shared Memory Accesses 

The shared memory accesses by `tcgen05.mma` and `tcgen05.cp` operations are performed
in the asynchronous proxy (async proxy).

Accessing the same memory location across miltiple proxies needs a cross-proxy fence.
For the async proxy, `fence.proxy.async` should be used to synchronize memory between
generic proxy and the async proxy.