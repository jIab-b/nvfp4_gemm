## 8.6. Proxies 

A *memory proxy*, or a *proxy* is an abstract label applied to a method of memory access. When two
memory operations use distinct methods of memory access, they are said to be different *proxies*.

Memory operations as defined in [Operation types](#operation-types) use *generic*
method of memory access, i.e. a *generic proxy*. Other operations such as textures and surfaces all
use distinct methods of memory access, also distinct from the *generic* method.

A *proxy fence* is required to synchronize memory operations across different *proxies*. Although
virtual aliases use the *generic* method of memory access, since using distinct virtual addresses
behaves as if using different *proxies*, they require a *proxy fence* to establish memory ordering.