##### 9.7.16.10.5. Data Path Layout Organization 

Different MMA variants access the tensor memory with different layout organization.
The following table lists the various layouts:

| M | cta\_group | A-Sparsity | Is .ws mode | Datapath organization | Layout ID | Tensor Memory Datapath Lane Alignment |
| --- | --- | --- | --- | --- | --- | --- |
| 32 | ::1 | Either | Yes | 1x4 | [Layout G](#tcgen05-data-path-layout-g) | 0 |
| 64 | ::1 | Either | Yes | 2x3 | [Layout E](#tcgen05-data-path-layout-e) | 0 |
| 64 | ::1 | Either | No | 4x1 (1/2 datapath utilized) | [Layout F](#tcgen05-data-path-layout-f) | 0 or 16 |
| 128 | ::1 | Either | Either | 4x1 | [Layout D](#tcgen05-data-path-layout-d) | 0 |
| 128 | ::2 | Dense | N/A | 2x2 | [Layout B](#tcgen05-data-path-layout-b) | 0 |
| 128 | ::2 | Sparse | N/A | 4x1 (1/2 datapath utilized) | [Layout C](#tcgen05-data-path-layout-c) | 0 or 16 |
| 256 | ::2 | Either | N/A | 4x1 | [Layout A](#tcgen05-data-path-layout-a) | 0 |

The layouts which utilize only half the datapath lanes, i.e.,
[Layout F](#tcgen05-data-path-layout-f) and
[Layout C](#tcgen05-data-path-layout-c), must use the same Tensor Memory
lane alignment across matrices `A`, `D` and the sparsity metadata matrix.

The following shows the warps that can access the Tensor Memory regions via
`tcgen05.ld` / `tcgen05.st` along with the addresses for various Tensor Memory Layouts.

###### 9.7.16.10.5.1. [Layout A (M = 256)](#tcgen05-data-path-layout-a)[](#tcgen05-data-path-layout-a "Permalink to this headline")

Layout organization for M = 256 is shown in [Figure 205](#tcgen05-data-path-layout-a1).

![_images/tcgen05-data-path-layout-a1.png](_images/tcgen05-data-path-layout-a1.png)


Figure 205 Layout organization for M = 256[](#tcgen05-data-path-layout-a1 "Permalink to this image")

Addresses for the above region to be used in `tcgen05.ld` / `tcgen05.st`
is shown in [Figure 206](#tcgen05-data-path-layout-a2)

![_images/tcgen05-data-path-layout-a2.png](_images/tcgen05-data-path-layout-a2.png)


Figure 206 Addresses to use in `tcgen05.ld` / `tcgen05.st`[](#tcgen05-data-path-layout-a2 "Permalink to this image")

###### 9.7.16.10.5.2. [Layout B (M = 128 + cta-group::2 + Dense A matrix)](#tcgen05-data-path-layout-b)[](#tcgen05-data-path-layout-b "Permalink to this headline")

Layout organization for M = 128 + .cta\_group::2 + Dense A matrix is shown in
[Figure 207](#tcgen05-data-path-layout-b1).

![_images/tcgen05-data-path-layout-b1.png](_images/tcgen05-data-path-layout-b1.png)


Figure 207 Layout organization for M = 128 + .cta\_group::2 + Dense A matrix[](#tcgen05-data-path-layout-b1 "Permalink to this image")

Addresses for the above region to be used in `tcgen05.ld` / `tcgen05.st`
is shown in [Figure 208](#tcgen05-data-path-layout-b2)

![_images/tcgen05-data-path-layout-b2.png](_images/tcgen05-data-path-layout-b2.png)


Figure 208 Addresses to use in `tcgen05.ld` / `tcgen05.st`[](#tcgen05-data-path-layout-b2 "Permalink to this image")

###### 9.7.16.10.5.3. [Layout C (M = 128 + cta-group::2 + Sparse A matrix)](#tcgen05-data-path-layout-c)[](#tcgen05-data-path-layout-c "Permalink to this headline")

Layout organization for M = 128 + .cta\_group::2 + Sparse A matrix is shown in
[Figure 209](#tcgen05-data-path-layout-c1).

![_images/tcgen05-data-path-layout-c1.png](_images/tcgen05-data-path-layout-c1.png)


Figure 209 Layout organization for M = 128 + .cta\_group::2 + Sparse A matrix[](#tcgen05-data-path-layout-c1 "Permalink to this image")

Addresses for the above region to be used in `tcgen05.ld` / `tcgen05.st`
is shown in [Figure 210](#tcgen05-data-path-layout-c2)

![_images/tcgen05-data-path-layout-c2.png](_images/tcgen05-data-path-layout-c2.png)


Figure 210 Addresses to use in `tcgen05.ld` / `tcgen05.st`[](#tcgen05-data-path-layout-c2 "Permalink to this image")

###### 9.7.16.10.5.4. [Layout D (M = 128 + cta-group::1)](#tcgen05-data-path-layout-d)[](#tcgen05-data-path-layout-d "Permalink to this headline")

Layout organization for M = 128 + .cta\_group::1 is shown in
[Figure 211](#tcgen05-data-path-layout-d1).

![_images/tcgen05-data-path-layout-d1.png](_images/tcgen05-data-path-layout-d1.png)


Figure 211 Layout organization for M = 128 + .cta\_group::1[](#tcgen05-data-path-layout-d1 "Permalink to this image")

Addresses for the above region to be used in `tcgen05.ld` / `tcgen05.st`
is shown in [Figure 212](#tcgen05-data-path-layout-d2)

![_images/tcgen05-data-path-layout-d2.png](_images/tcgen05-data-path-layout-d2.png)


Figure 212 Addresses to use in `tcgen05.ld` / `tcgen05.st`[](#tcgen05-data-path-layout-d2 "Permalink to this image")

###### 9.7.16.10.5.5. [Layout E (M = 64 + .ws mode)](#tcgen05-data-path-layout-e)[](#tcgen05-data-path-layout-e "Permalink to this headline")

Layout organization for M = 64 + .ws mode is shown in
[Figure 213](#tcgen05-data-path-layout-e1).

![_images/tcgen05-data-path-layout-e1.png](_images/tcgen05-data-path-layout-e1.png)


Figure 213 Layout organization for M = 64 + .ws mode[](#tcgen05-data-path-layout-e1 "Permalink to this image")

Addresses for the above region to be used in `tcgen05.ld` / `tcgen05.st`
is shown in [Figure 214](#tcgen05-data-path-layout-e2)

![_images/tcgen05-data-path-layout-e2.png](_images/tcgen05-data-path-layout-e2.png)


Figure 214 Addresses to use in `tcgen05.ld` / `tcgen05.st`[](#tcgen05-data-path-layout-e2 "Permalink to this image")

###### 9.7.16.10.5.6. [Layout F (M = 64 + non .ws mode)](#tcgen05-data-path-layout-f)[](#tcgen05-data-path-layout-f "Permalink to this headline")

Layout organization for M = 64 + non .ws mode is shown in
[Figure 215](#tcgen05-data-path-layout-f1).

![_images/tcgen05-data-path-layout-f1.png](_images/tcgen05-data-path-layout-f1.png)


Figure 215 Layout organization for M = 64 + non .ws mode[](#tcgen05-data-path-layout-f1 "Permalink to this image")

Addresses for the above region to be used in `tcgen05.ld` / `tcgen05.st`
is shown in [Figure 216](#tcgen05-data-path-layout-f2)

![_images/tcgen05-data-path-layout-f2.png](_images/tcgen05-data-path-layout-f2.png)


Figure 216 Addresses to use in `tcgen05.ld` / `tcgen05.st`[](#tcgen05-data-path-layout-f2 "Permalink to this image")

###### 9.7.16.10.5.7. [Layout G (M = 32)](#tcgen05-data-path-layout-g)[](#tcgen05-data-path-layout-g "Permalink to this headline")

Layout organization for M = 32 is shown in
[Figure 217](#tcgen05-data-path-layout-g1).

![_images/tcgen05-data-path-layout-g1.png](_images/tcgen05-data-path-layout-g1.png)


Figure 217 Layout organization for M = 32[](#tcgen05-data-path-layout-g1 "Permalink to this image")

Addresses for the above region to be used in `tcgen05.ld` / `tcgen05.st`
is shown in [Figure 218](#tcgen05-data-path-layout-g2)

![_images/tcgen05-data-path-layout-g2.png](_images/tcgen05-data-path-layout-g2.png)


Figure 218 Addresses to use in `tcgen05.ld` / `tcgen05.st`[](#tcgen05-data-path-layout-g2 "Permalink to this image")