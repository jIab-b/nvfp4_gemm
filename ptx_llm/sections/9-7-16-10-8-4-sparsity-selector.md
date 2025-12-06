###### 9.7.16.10.8.4. Sparsity selector 

The value of the sparsity selector selects the sub-columns in the Tensor Memory
to form the sparsity metadata matrix, which is used with matrix `A` to form the
multiplicand matrix.

The following shows the sparse metadata matrix layout in Tensor Memory for various MMA variants:

###### 9.7.16.10.8.4.1. [Layout of the Sparsity Metadata Matrix for M = 64 for `.kind::f16`](#tcgen05-sparse-matrices-sparsity-selector-kind-f16-m64)[](#tcgen05-sparse-matrices-sparsity-selector-kind-f16-m64 "Permalink to this headline")

[Figure 262](#tcgen05-sparse-matrices-sparsity-selector-kind-f16-m64-dig) shows which sub-columns gets
selected for different values of Sparsity Selector.

![_images/tcgen05-sparse-matrices-sparsity-selector-kind-f16-m64.png](_images/tcgen05-sparse-matrices-sparsity-selector-kind-f16-m64.png)


Figure 262 Sparsity Metadata Layout for M = 64 for `.kind::f16`[](#tcgen05-sparse-matrices-sparsity-selector-kind-f16-m64-dig "Permalink to this image")

###### 9.7.16.10.8.4.2. [Layout of the Sparsity Metadata Matrix for M = 128 / M = 256 for `.kind::f16`](#tcgen05-sparse-matrices-sparsity-selector-kind-f16-m128-256)[](#tcgen05-sparse-matrices-sparsity-selector-kind-f16-m128-256 "Permalink to this headline")

[Figure 263](#tcgen05-sparse-matrices-sparsity-selector-kind-f16-m128-256-dig) shows which sub-columns gets
selected for different values of Sparsity Selector.

![_images/tcgen05-sparse-matrices-sparsity-selector-kind-f16-m128-256.png](_images/tcgen05-sparse-matrices-sparsity-selector-kind-f16-m128-256.png)


Figure 263 Sparsity Metadata Layout for M = 128 / M = 256 for `.kind::f16`[](#tcgen05-sparse-matrices-sparsity-selector-kind-f16-m128-256-dig "Permalink to this image")

###### 9.7.16.10.8.4.3. [Layout of the Sparsity Metadata Matrix for M = 64 for `.kind::tf32`](#tcgen05-sparse-matrices-sparsity-selector-kind-tf32-m64)[](#tcgen05-sparse-matrices-sparsity-selector-kind-tf32-m64 "Permalink to this headline")

[Figure 264](#tcgen05-sparse-matrices-sparsity-selector-kind-tf32-m64-dig) shows which sub-columns gets
selected for different values of Sparsity Selector.

![_images/tcgen05-sparse-matrices-sparsity-selector-kind-tf32-m64.png](_images/tcgen05-sparse-matrices-sparsity-selector-kind-tf32-m64.png)


Figure 264 Sparsity Metadata Layout for M = 64 for `.kind::tf32`[](#tcgen05-sparse-matrices-sparsity-selector-kind-tf32-m64-dig "Permalink to this image")

###### 9.7.16.10.8.4.4. [Layout of the Sparsity Metadata Matrix for M = 128 / M = 256 for `.kind::tf32`](#tcgen05-sparse-matrices-sparsity-selector-kind-tf32-m128-256)[](#tcgen05-sparse-matrices-sparsity-selector-kind-tf32-m128-256 "Permalink to this headline")

[Figure 265](#tcgen05-sparse-matrices-sparsity-selector-kind-tf32-m128-256-dig) shows which sub-columns gets
selected for different values of Sparsity Selector.

![_images/tcgen05-sparse-matrices-sparsity-selector-kind-tf32-m128-256.png](_images/tcgen05-sparse-matrices-sparsity-selector-kind-tf32-m128-256.png)


Figure 265 Sparsity Metadata Layout for M = 128 / M = 256 for `.kind::tf32`[](#tcgen05-sparse-matrices-sparsity-selector-kind-tf32-m128-256-dig "Permalink to this image")

###### 9.7.16.10.8.4.5. [Layout of the Sparsity Metadata Matrix for M = 64 for `.kind::f8f6f4`, `.kind::mxf8f6f4`, `.kind::i8`, `.kind::mxf4`, `.kind::mxf4nvf4`](#tcgen05-sparse-matrices-sparsity-selector-kind-f8f6f4-mxf8f6f4-m64)[](#tcgen05-sparse-matrices-sparsity-selector-kind-f8f6f4-mxf8f6f4-m64 "Permalink to this headline")

The value of the sparsity selector:

* must be 0 for `.kind::i8` and `.kind::f8f6f4`
* is assumed to be 0 for `.kind::mxf8f6f4`, `.kind::mxf4` and `.kind::mxf4nvf4`

and all of the columns are selected as
shown in [Figure 266](#tcgen05-sparse-matrices-sparsity-selector-kind-f8f6f4-mxf8f6f4-m64-dig)

![_images/tcgen05-sparse-matrices-sparsity-selector-kind-f8f6f4-mxf8f6f4-m64.png](_images/tcgen05-sparse-matrices-sparsity-selector-kind-f8f6f4-mxf8f6f4-m64.png)


Figure 266 Sparsity Metadata Layout for M = 64 for `.kind::f8f6f4`, `.kind::mxf8f6f4`, `.kind::i8`, `.kind::mxf4`, `.kind::mxf4nvf4`[](#tcgen05-sparse-matrices-sparsity-selector-kind-f8f6f4-mxf8f6f4-m64-dig "Permalink to this image")

###### 9.7.16.10.8.4.6. [Layout of the Sparsity Metadata Matrix for M = 128 / M = 256 for `.kind::f8f6f4`, `.kind::mxf8f6f4`, `.kind::i8`, `.kind::mxf4`, `.kind::mxf4nvf4`](#tcgen05-sparse-matrices-sparsity-selector-kind-f8f6f4-mxf8f6f4-m128-256)[](#tcgen05-sparse-matrices-sparsity-selector-kind-f8f6f4-mxf8f6f4-m128-256 "Permalink to this headline")

The value of the sparsity selector:

* must be 0 for `.kind::i8` and `.kind::f8f6f4`
* is assumed to be 0 for `.kind::mxf8f6f4`, `.kind::mxf4` and `.kind::mxf4nvf4`

and all of the columns are selected as
shown in [Figure 267](#tcgen05-sparse-matrices-sparsity-selector-kind-f8f6f4-mxf8f6f4-m128-256-dig)

![_images/tcgen05-sparse-matrices-sparsity-selector-kind-f8f6f4-mxf8f6f4-m128-256.png](_images/tcgen05-sparse-matrices-sparsity-selector-kind-f8f6f4-mxf8f6f4-m128-256.png)


Figure 267 Sparsity Metadata Layout for M = 128 / M = 256 for `.kind::f8f6f4`, `.kind::mxf8f6f4`, `.kind::i8`, `.kind::mxf4`, `.kind::mxf4nvf4`[](#tcgen05-sparse-matrices-sparsity-selector-kind-f8f6f4-mxf8f6f4-m128-256-dig "Permalink to this image")