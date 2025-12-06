### 5.5.8. Tensor-map 

The tensor-map is a 128-byte opaque object either in `.const` space or `.param` (kernel function
parameter) space or `.global` space which describes the tensor properties and the access properties
of the tensor data described in previous sections.

Tensor-Map can be created using CUDA APIs. Refer to *CUDA programming guide* for more details.