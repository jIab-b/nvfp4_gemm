### 8.1.1. Limitations on atomicity at system scope 

When communicating with the host CPU, certain strong operations with system scope may not be
performed atomically on some systems. For more details on atomicity guarantees to host memory, see
the *CUDA Atomicity Requirements*.