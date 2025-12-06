##### 9.7.16.2.3. Data Movement Shape 

The data movement shape indicates the dimension of the data to be moved to or from the
[Tensor Memory](#tensor-memory). These shapes are described as a tuple `lane x size` where:

* `lane` indicates the number of rows in the [Tensor Memory](#tensor-memory); and
* `size` indicates the amount of data, in units of bits (b), across the columns in the
  [Tensor Memory](#tensor-memory).

The following shapes are supported by various tcgen05 operations:

| Shape | tcgen05.<op> |
| --- | --- |
| `.16x64b`, `.16x128b`, `.16x256b`, `.16x32bx2`, `.32x32b` | `.ld` / `.st` |
| `.4x256b`, `.32x128b`, `.64x128b`, `.128x256b`, `.128x128b` | `.cp` |
| `.31x256b` (implicit) | `.shift` |

###### 9.7.16.2.3.1. [Memory Layout](#tcgen05-memory-layout)[](#tcgen05-memory-layout "Permalink to this headline")

The following shows the layout of the matrix fragments across threads of the warp.

###### 9.7.16.2.3.1.1. [Matrix fragments for shape .32x32b](#tcgen05-matrix-fragments-shape-3232b)[](#tcgen05-matrix-fragments-shape-3232b "Permalink to this headline")

A `tcgen05{.ld,.st}.32x32b` instruction has the following data vector register.

| Fragment | Elements (low to high) |
| --- | --- |
| A vector expression containing `.num` number of `.b32` registers as mentioned in the [Table 47](#tcgen05-num-shapes-ld). | r0, r1, … |

A warp executing `tcgen05{.ld,.st}.32x32b` will access 32 lanes of the Tensor Memory.
It loads from or stores to each of the lane (32 \* .num)-bits of data as shown in
[Figure 183](#tcgen05-mma-fragment-3232b).

![_images/tcgen05-mma-fragment-3232b.png](_images/tcgen05-mma-fragment-3232b.png)


Figure 183 Matrix Fragment for shape .32x32b[](#tcgen05-mma-fragment-3232b "Permalink to this image")

###### 9.7.16.2.3.1.2. [Matrix fragments for shape .16x64b](#tcgen05-matrix-fragments-shape-6464b)[](#tcgen05-matrix-fragments-shape-6464b "Permalink to this headline")

A `tcgen05{.ld,.st}.16x64b` instruction has the following data vector register.

| Fragment | Elements (low to high) |
| --- | --- |
| A vector expression containing `.num` number of `.b32` registers as mentioned in the [Table 47](#tcgen05-num-shapes-ld). | r0, r1, … |

A warp executing `tcgen05{.ld,.st}.16x64b` will access 16 lanes of the Tensor Memory.
It loads from or stores to each of the lane (64 \* .num)-bits of data as shown in
[Figure 184](#tcgen05-mma-fragment-1664b).

![_images/tcgen05-mma-fragment-1664b.png](_images/tcgen05-mma-fragment-1664b.png)


Figure 184 Matrix Fragment for shape .16x64b[](#tcgen05-mma-fragment-1664b "Permalink to this image")

###### 9.7.16.2.3.1.3. [Matrix fragments for shape .16x128b](#tcgen05-matrix-fragments-shape-16128b)[](#tcgen05-matrix-fragments-shape-16128b "Permalink to this headline")

A `tcgen05{.ld,.st}.16x128b` instruction has the following data vector register.

| Fragment | Elements (low to high) |
| --- | --- |
| A vector expression containing `.num` number of `.b32` registers as mentioned in the [Table 47](#tcgen05-num-shapes-ld). | r0, r1, … |

A warp executing `tcgen05{.ld,.st}.16x128b` will access 16 lanes of the Tensor Memory.
It loads from or stores to each of the lane (128 \* .num)-bits of data as shown in
[Figure 185](#tcgen05-mma-fragment-16128b).

![_images/tcgen05-mma-fragment-16128b.png](_images/tcgen05-mma-fragment-16128b.png)


Figure 185 Matrix Fragment for shape .16x128b[](#tcgen05-mma-fragment-16128b "Permalink to this image")

###### 9.7.16.2.3.1.4. [Matrix fragments for shape .16x256b](#tcgen05-matrix-fragments-shape-16256b)[](#tcgen05-matrix-fragments-shape-16256b "Permalink to this headline")

A `tcgen05{.ld,.st}.16x256b` instruction has the following data vector register.

| Fragment | Elements (low to high) |
| --- | --- |
| A vector expression containing `.num` number of `.b32` registers as mentioned in the [Table 47](#tcgen05-num-shapes-ld). | r0, r1, r2, r3, … |

A warp executing `tcgen05{.ld,.st}.16x256b` will access 16 lanes of the Tensor Memory.
It loads from or stores to each of the lane (256 \* .num)-bits of data as shown in
[Figure 186](#tcgen05-mma-fragment-16256b).

![_images/tcgen05-mma-fragment-16256b.png](_images/tcgen05-mma-fragment-16256b.png)


Figure 186 Matrix Fragment for shape .16x256b[](#tcgen05-mma-fragment-16256b "Permalink to this image")

###### 9.7.16.2.3.1.5. [Matrix fragments for shape .16x32bx2](#tcgen05-matrix-fragments-shape-1632b2)[](#tcgen05-matrix-fragments-shape-1632b2 "Permalink to this headline")

A `tcgen05{.ld,.st}.16x32bx2` instruction has the following data vector register.

| Fragment | Elements (low to high) |
| --- | --- |
| A vector expression containing `.num` number of `.b32` registers as mentioned in the [Table 47](#tcgen05-num-shapes-ld). | r0, r1, … |

A warp executing `tcgen05{.ld,.st}.16x32bx2` will access 16 lanes of the Tensor Memory.
It loads from or stores to each of the lane (32 \* .num)-bits of data as shown in
[Figure 187](#tcgen05-mma-fragment-1632b2).

![_images/tcgen05-mma-fragment-1632b2.png](_images/tcgen05-mma-fragment-1632b2.png)


Figure 187 Matrix Fragment for shape .16x32bx2[](#tcgen05-mma-fragment-1632b2 "Permalink to this image")