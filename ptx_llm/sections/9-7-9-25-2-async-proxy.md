##### 9.7.9.25.2. Async Proxy 

The `cp{.reduce}.async.bulk` operations are performed in the *asynchronous proxy* (or *async
proxy*).

Accessing the same memory location across multiple proxies needs a cross-proxy fence. For the
*async proxy*, `fence.proxy.async` should be used to synchronize memory between *generic
proxy* and the *async proxy*.

The completion of a `cp{.reduce}.async.bulk` operation is followed by an implicit *generic-async*
proxy fence. So the result of the asynchronous operation is made visible to the generic proxy as
soon as its completion is observed. *Async-group* OR *mbarrier-based* completion mechanism must
be used to wait for the completion of the `cp{.reduce}.async.bulk` instructions.