##### 9.7.16.10.7. Block Scaling 

The `tcgen05.mma` instructions with the following `.kind` qualifier:

* `.kind::mxf8f6f4`
* `.kind::mxf4`
* `.kind::mxf4nvf4`

perform matrix multiplication with block scaling. This operation has the following form:

`(A * scale_A)  * (B * scale_B) + D`

where `scale_A` and `scale_B` are matrices residing in [Tensor Memory](#tensor-memory).

For a `scale_A` matrix of shape *M x SFA\_N*, each row of matrix `A` is divided into
*SFA\_N* number of chunks and each chunk of a row is multiplied with the corresponding
element in the *SF\_A* of the same row.

Similarly, for a `scale_B` matrix of shape *SFB\_M x N*, each column of matrix `B` is
divided into the *SFB\_M* number of chunks and each chunk of a column is multiplied with
the corresponding element in the *SF\_B* of the same column.

Scale factors for `A` and `B` matrices need to be duplicated to all 32 lane partitions
of tensor memory.

[Figure 230](#tcgen05-mma-block-scaling) shows an example of `tcgen05.mma` with block scaling of
`scale_vec::2X`.

![_images/tcgen05-mma-block-scaling.png](_images/tcgen05-mma-block-scaling.png)


Figure 230 `tcgen05.mma` with block scaling of `scale_vec::2X`[](#tcgen05-mma-block-scaling "Permalink to this image")

###### 9.7.16.10.7.1. [Valid combinations of scale\_vectorsize with types and MMA-Kind](#tcgen05-mma-scale-valid-vec-size)[](#tcgen05-mma-scale-valid-vec-size "Permalink to this headline")

The shape of *scale\_A* and *scale\_B* matrices depend on the `.scale_vectorsize` as shown in
[Table 54](#tcgen05-mma-scale-valid-comb).

Table 54 Valid combinations of scale\_vectorsize and shapes[](#tcgen05-mma-scale-valid-comb "Permalink to this table")







| .scale\_vectorsize | .kind::\* | K | Shape of scale\_A | Shape of scale\_B |
| --- | --- | --- | --- | --- |
| `.scale_vec::1X` | `.kind::mxf8f6f4` | All supported values of K | M x 1 | 1 x N |
| `.scale_vec::2X` | `.kind::mxf4`, `.kind::mxf4nvf4` | All supported values of K | M x 2 | 2 x N |
| `.scale_vec::4X` | `.kind::mxf4nvf4` | All supported values of K | M x 4 | 4 x N |
| `.block16` | `.kind::mxf4nvf4` | K = 96 | M x 6 | 6 x N |
| All supported values of K except 96 | M x 4 | 4 x N |
| `.block32` | `.kind::mxf4`, `.kind::mxf4nvf4` | K = 96 | M x 3 | 3 x N |
| All supported values of K except 96 | M x 2 | 2 x N |
| `.kind::mxf8f6f4` | All supported values of K | M x 1 | 1 x N |

The valid combination of the exact element types and the `.scale_vectorsize` are listed in
[Table 55](#tcgen05-mma-scale-valid-comb-detail).

Table 55 Valid combinations of scale\_vectorsize with types and MMA-Kind[](#tcgen05-mma-scale-valid-comb-detail "Permalink to this table")






| .kind::\* | Element Data Type | Scale Data Type | .scale\_vectorsize |
| --- | --- | --- | --- |
| `.kind::mxf8f6f4` | E4M3, E5M2, E2M3 E3M2, E2M1 | UE8M0 | `.scale_vec::1X` / `.block32` |
| `.kind::mxf4` | E2M1 | UE8M0 | `.scale_vec::2X` / `.block32` |
| `.kind::mxf4nvf4` | E2M1 | UE8M0 | `.scale_vec::2X` / `.block32`, `.scale_vec::4X` / `.block16` |
| E2M1 | UE4M3 | `.scale_vec::4X` / `.block16` |

New `.blockN` qualifiers are aliases for `.scale_vec::NX` qualifiers as:

* `.block32` is alias for `.scale_vec::1X` or `.scale_vec::2X`
  based on `.kind` and K dimension
* `.block16` is alias for `.scale_vec::4X`

###### 9.7.16.10.7.2. [Scale Factor A ID](#tcgen05-mma-scale-factor-a)[](#tcgen05-mma-scale-factor-a "Permalink to this headline")

The value of the scale factor `A ID` selects the sub-columns in the Tensor Memory to
form the scale factor `A` matrix, which is used to scale the matrix `A`.

The following shows the scale factor matrix layout for various scale vector sizes:

###### 9.7.16.10.7.2.1. [Layout of the Scale Factor A Matrix for scale\_vec::1X/block32 with K=32/K=64](#tcgen05-mma-scale-factor-a-layout-1x)[](#tcgen05-mma-scale-factor-a-layout-1x "Permalink to this headline")

There is one scale factor per row of the `A` matrix with block size as 32 and the scale factor must be provided in
1-byte aligned sub-column of the Tensor Memory. *SFA\_ID* specifies the byte offset in the
Tensor Memory word that must be used for the scale factor matrix.
[Figure 231](#tcgen05-mma-scale-factor-a-1x-dig) shows which sub-columns get selected for
different values of *SFA\_ID*.

![_images/tcgen05-mma-scale-factor-a-1x-dig.png](_images/tcgen05-mma-scale-factor-a-1x-dig.png)


Figure 231 Layout of scale factor A matrix with scale\_vec::1X/block32 with K=32/K=64[](#tcgen05-mma-scale-factor-a-1x-dig "Permalink to this image")

For example, if *SFA\_ID* is 0, then all the green columns are selected to form the scale factor
matrix. Similarly, *SFA\_ID* values of 1, 2 and 3 would select the blue, yellow, and red columns,
respectively.

###### 9.7.16.10.7.2.2. [Layout of the Scale Factor A Matrix for scale\_vec::2X/block32 with K=64/K=128](#tcgen05-mma-scale-factor-a-layout-2x)[](#tcgen05-mma-scale-factor-a-layout-2x "Permalink to this headline")

There are two scale factors per row of the `A` matrix with block size as 32 and the scale factor must be provided in
2-byte aligned sub-column of the Tensor Memory. *SFA\_ID* specifies the half word offset in the
Tensor Memory word that must be used for the scale factor matrix.
[Figure 232](#tcgen05-mma-scale-factor-a-2x-dig) shows which sub-columns gets selected for different
values of *SFA\_ID*.

![_images/tcgen05-mma-scale-factor-a-2x-dig.png](_images/tcgen05-mma-scale-factor-a-2x-dig.png)


Figure 232 Layout of scale factor A matrix with scale\_vec::2X/block32 with K=64/K=128[](#tcgen05-mma-scale-factor-a-2x-dig "Permalink to this image")

For example, if *SFA\_ID* is 0, then all the green columns are selected to form the scale factor
matrix. Similarly, if *SFA\_ID* is 2, then all of the blue columns are selected to form the scale
factor matrix.

###### 9.7.16.10.7.2.3. [Layout of the Scale Factor A Matrix for scale\_vec::4X/block16 with K=64/K=128](#tcgen05-mma-scale-factor-a-layout-4x)[](#tcgen05-mma-scale-factor-a-layout-4x "Permalink to this headline")

There are four scale factors per row of the `A` matrix with block size as 16 and the scale factor must be provided in
4-byte aligned sub-column of the Tensor Memory. The *SFA\_ID* value must be 0 and this specifies
that all of the columns (in green) will be used for the scale factor matrix.
[Figure 233](#tcgen05-mma-scale-factor-a-4x-dig) shows which sub-columns gets selected for different
values of *SFA\_ID*.

![_images/tcgen05-mma-scale-factor-a-4x-dig.png](_images/tcgen05-mma-scale-factor-a-4x-dig.png)


Figure 233 Layout of scale factor A matrix with scale\_vec::4X/block16 with K=64/K=128[](#tcgen05-mma-scale-factor-a-4x-dig "Permalink to this image")

###### 9.7.16.10.7.2.4. [Layout of the Scale Factor A Matrix for block32 with K=96 (Semantically equivalent to scale\_vec::3X)](#tcgen05-mma-scale-factor-a-layout-block32-k96)[](#tcgen05-mma-scale-factor-a-layout-block32-k96 "Permalink to this headline")

There are three scale factors per row of the `A` matrix with block size as 32 and the scale
factor must be provided in 4-byte aligned sub-column of the Tensor Memory. *SFA\_ID* specifies
the byte offset in the Tensor Memory word that must be used for the scale factor matrix.
[Figure 234](#tcgen05-mma-scale-factor-a-block32-k96-dig1), [Figure 235](#tcgen05-mma-scale-factor-a-block32-k96-dig2),
[Figure 236](#tcgen05-mma-scale-factor-a-block32-k96-dig3) and [Figure 237](#tcgen05-mma-scale-factor-a-block32-k96-dig4)
show which sub-columns get selected for different values of *SFA\_ID*.

![_images/tcgen05-mma-scale-factor-a-block32-k96-dig1.png](_images/tcgen05-mma-scale-factor-a-block32-k96-dig1.png)


Figure 234 Layout of scale factor A matrix with block32 with K=96 with SFA\_ID=00[](#tcgen05-mma-scale-factor-a-block32-k96-dig1 "Permalink to this image")


![_images/tcgen05-mma-scale-factor-a-block32-k96-dig2.png](_images/tcgen05-mma-scale-factor-a-block32-k96-dig2.png)


Figure 235 Layout of scale factor A matrix with block32 with K=96 with SFA\_ID=01[](#tcgen05-mma-scale-factor-a-block32-k96-dig2 "Permalink to this image")


![_images/tcgen05-mma-scale-factor-a-block32-k96-dig3.png](_images/tcgen05-mma-scale-factor-a-block32-k96-dig3.png)


Figure 236 Layout of scale factor A matrix with block32 with K=96 with SFA\_ID=10[](#tcgen05-mma-scale-factor-a-block32-k96-dig3 "Permalink to this image")


![_images/tcgen05-mma-scale-factor-a-block32-k96-dig4.png](_images/tcgen05-mma-scale-factor-a-block32-k96-dig4.png)


Figure 237 Layout of scale factor A matrix with block32 with K=96 with SFA\_ID=11[](#tcgen05-mma-scale-factor-a-block32-k96-dig4 "Permalink to this image")

For example, if *SFA\_ID* is 0, then all the green columns are selected to form the scale factor
matrix. Similarly, *SFA\_ID* values of 1, 2 and 3 would select the blue, yellow, and red columns,
respectively.

###### 9.7.16.10.7.2.5. [Layout of the Scale Factor A Matrix for block16 with K=96 (Semantically equivalent to scale\_vec::6X)](#tcgen05-mma-scale-factor-a-layout-block16-k96)[](#tcgen05-mma-scale-factor-a-layout-block16-k96 "Permalink to this headline")

There are six scale factors per row of the `A` matrix with block size as 16 and the scale
factor must be provided in 4-byte aligned sub-column of the Tensor Memory. *SFA\_ID* specifies
the byte offset in the Tensor Memory word that must be used for the scale factor matrix.
[Figure 238](#tcgen05-mma-scale-factor-a-block16-k96-dig1) and [Figure 239](#tcgen05-mma-scale-factor-a-block16-k96-dig2)
show which sub-columns get selected for different values of *SFA\_ID*.

![_images/tcgen05-mma-scale-factor-a-block16-k96-dig1.png](_images/tcgen05-mma-scale-factor-a-block16-k96-dig1.png)


Figure 238 Layout of scale factor A matrix with block16 with K=96 with SFA\_ID=00[](#tcgen05-mma-scale-factor-a-block16-k96-dig1 "Permalink to this image")


![_images/tcgen05-mma-scale-factor-a-block16-k96-dig2.png](_images/tcgen05-mma-scale-factor-a-block16-k96-dig2.png)


Figure 239 Layout of scale factor A matrix with block16 with K=96 with SFA\_ID=10[](#tcgen05-mma-scale-factor-a-block16-k96-dig2 "Permalink to this image")

For example, if *SFA\_ID* is 0, then all the green columns are selected to form the scale factor
matrix. Similarly, if *SFA\_ID* is 2, then all of the blue columns are selected to form the scale
factor matrix.

###### 9.7.16.10.7.3. [Scale Factor B ID](#tcgen05-mma-scale-factor-b)[](#tcgen05-mma-scale-factor-b "Permalink to this headline")

The value of the scale factor `B ID` selects the sub-columns in the Tensor Memory to
form the scale factor `B` matrix, which is used to scale the matrix `B`.

The following shows the scale factor matrix layout for various scale vector sizes:

###### 9.7.16.10.7.3.1. [Layout of the Scale Factor B Matrix for scale\_vec::1X/block32 with K=32/K=64](#tcgen05-mma-scale-factor-b-layout-1x)[](#tcgen05-mma-scale-factor-b-layout-1x "Permalink to this headline")

There is one scale factor per row of the `B` matrix with block size as 32 and the scale factor must be provided in
1-byte aligned sub-column of the Tensor Memory. *SFB\_ID* specifies the byte offset in the
Tensor Memory word that must be used for the scale factor matrix.
[Figure 240](#tcgen05-mma-scale-factor-b-1x-dig) shows which sub-columns get selected for
different values of *SFB\_ID*.

![_images/tcgen05-mma-scale-factor-b-1x-dig.png](_images/tcgen05-mma-scale-factor-b-1x-dig.png)


Figure 240 Layout of scale factor B matrix with scale\_vec::1X/block32 with K=32/K=64[](#tcgen05-mma-scale-factor-b-1x-dig "Permalink to this image")

For example, if *SFB\_ID* is 0, then all the green columns are selected to form the scale factor
matrix. Similarly, *SFB\_ID* values of 1, 2 and 3 would select the blue, yellow, and red columns, respectively.

###### 9.7.16.10.7.3.2. [Layout of the Scale Factor B Matrix for scale\_vec::2X/block32 with K=64/K=128](#tcgen05-mma-scale-factor-b-layout-2x)[](#tcgen05-mma-scale-factor-b-layout-2x "Permalink to this headline")

There are two scale factors per row of the `B` matrix with block size as 32 and the scale factor must be provided in
2-byte aligned sub-column of the Tensor Memory. *SFB\_ID* specifies the half word offset in the
Tensor Memory word that must be used for the scale factor matrix.
[Figure 241](#tcgen05-mma-scale-factor-b-2x-dig) shows which sub-columns get selected for
different values of *SFB\_ID*.

![_images/tcgen05-mma-scale-factor-b-2x-dig.png](_images/tcgen05-mma-scale-factor-b-2x-dig.png)


Figure 241 Layout of scale factor B matrix with scale\_vec::2X/block32 with K=64/K=128[](#tcgen05-mma-scale-factor-b-2x-dig "Permalink to this image")

For example, if *SFB\_ID* is 0, then all the green columns are selected to form the scale factor
matrix. Similarly, if *SFB\_ID* is 2, then all of the blue columns are selected to form the scale
factor matrix.

###### 9.7.16.10.7.3.3. [Layout of the Scale Factor B Matrix for scale\_vec::4X/block16 with K=64/K=128](#tcgen05-mma-scale-factor-b-layout-4x)[](#tcgen05-mma-scale-factor-b-layout-4x "Permalink to this headline")

There are four scale factors per row of the `B` matrix with block size as 16 and the scale factor must be provided in
4-byte aligned sub-column of the Tensor Memory. The *SFB\_ID* value must be 0 and this specifies
that all of the columns (in green) will be used for the scale factor matrix.
[Figure 242](#tcgen05-mma-scale-factor-b-4x-dig) shows which sub-columns get selected for
different values of *SFB\_ID*.

![_images/tcgen05-mma-scale-factor-b-4x-dig.png](_images/tcgen05-mma-scale-factor-b-4x-dig.png)


Figure 242 Layout of scale factor B matrix with scale\_vec::4X/block16 with K=64/K=128[](#tcgen05-mma-scale-factor-b-4x-dig "Permalink to this image")

###### 9.7.16.10.7.3.4. [Layout of the Scale Factor B Matrix for block32 with K=96 (Semantically equivalent to scale\_vec::3X)](#tcgen05-mma-scale-factor-b-layout-block32-k96)[](#tcgen05-mma-scale-factor-b-layout-block32-k96 "Permalink to this headline")

There are three scale factors per row of the `B` matrix with block size as 32 and the scale factor
must be provided in 4-byte aligned sub-column of the Tensor Memory. *SFB\_ID* specifies the byte
offset in the Tensor Memory word that must be used for the scale factor matrix.

For N<=128, [Figure 243](#tcgen05-mma-scale-factor-b-block32-k96-nlt128-dig1),
[Figure 244](#tcgen05-mma-scale-factor-b-block32-k96-nlt128-dig2),
[Figure 245](#tcgen05-mma-scale-factor-b-block32-k96-nlt128-dig3) and
[Figure 246](#tcgen05-mma-scale-factor-b-block32-k96-nlt128-dig4) show which
sub-columns get selected for different values of *SFB\_ID*.

![_images/tcgen05-mma-scale-factor-b-block32-k96-nlt128-dig1.png](_images/tcgen05-mma-scale-factor-b-block32-k96-nlt128-dig1.png)


Figure 243 Layout of scale factor B matrix with block32 with K=96 and N<=128 with SFA\_ID=00[](#tcgen05-mma-scale-factor-b-block32-k96-nlt128-dig1 "Permalink to this image")


![_images/tcgen05-mma-scale-factor-b-block32-k96-nlt128-dig2.png](_images/tcgen05-mma-scale-factor-b-block32-k96-nlt128-dig2.png)


Figure 244 Layout of scale factor B matrix with block32 with K=96 and N<=128 with SFA\_ID=01[](#tcgen05-mma-scale-factor-b-block32-k96-nlt128-dig2 "Permalink to this image")


![_images/tcgen05-mma-scale-factor-b-block32-k96-nlt128-dig3.png](_images/tcgen05-mma-scale-factor-b-block32-k96-nlt128-dig3.png)


Figure 245 Layout of scale factor B matrix with block32 with K=96 and N<=128 with SFA\_ID=10[](#tcgen05-mma-scale-factor-b-block32-k96-nlt128-dig3 "Permalink to this image")


![_images/tcgen05-mma-scale-factor-b-block32-k96-nlt128-dig4.png](_images/tcgen05-mma-scale-factor-b-block32-k96-nlt128-dig4.png)


Figure 246 Layout of scale factor B matrix with block32 with K=96 and N<=128 with SFA\_ID=11[](#tcgen05-mma-scale-factor-b-block32-k96-nlt128-dig4 "Permalink to this image")

For N>128, [Figure 247](#tcgen05-mma-scale-factor-b-block32-k96-ngt128-dig1),
[Figure 248](#tcgen05-mma-scale-factor-b-block32-k96-ngt128-dig2),
[Figure 249](#tcgen05-mma-scale-factor-b-block32-k96-ngt128-dig3),
[Figure 250](#tcgen05-mma-scale-factor-b-block32-k96-ngt128-dig4),
[Figure 251](#tcgen05-mma-scale-factor-b-block32-k96-ngt128-dig5) and
[Figure 252](#tcgen05-mma-scale-factor-b-block32-k96-ngt128-dig6) show which
sub-columns get selected for different values of *SFB\_ID*.

![_images/tcgen05-mma-scale-factor-b-block32-k96-ngt128-dig1.png](_images/tcgen05-mma-scale-factor-b-block32-k96-ngt128-dig1.png)


Figure 247 Layout of scale factor B matrix with block32 with K=96 and N>128 with SFA\_ID=00[](#tcgen05-mma-scale-factor-b-block32-k96-ngt128-dig1 "Permalink to this image")


![_images/tcgen05-mma-scale-factor-b-block32-k96-ngt128-dig2.png](_images/tcgen05-mma-scale-factor-b-block32-k96-ngt128-dig2.png)


Figure 248 Layout of scale factor B matrix with block32 with K=96 and N>128 with SFA\_ID=01[](#tcgen05-mma-scale-factor-b-block32-k96-ngt128-dig2 "Permalink to this image")


![_images/tcgen05-mma-scale-factor-b-block32-k96-ngt128-dig3.png](_images/tcgen05-mma-scale-factor-b-block32-k96-ngt128-dig3.png)


Figure 249 Layout of scale factor B matrix with block32 with K=96 and N>128 with SFA\_ID=10[](#tcgen05-mma-scale-factor-b-block32-k96-ngt128-dig3 "Permalink to this image")


![_images/tcgen05-mma-scale-factor-b-block32-k96-ngt128-dig4.png](_images/tcgen05-mma-scale-factor-b-block32-k96-ngt128-dig4.png)


Figure 250 Layout of scale factor B matrix with block32 with K=96 and N>128 with SFA\_ID=10[](#tcgen05-mma-scale-factor-b-block32-k96-ngt128-dig4 "Permalink to this image")


![_images/tcgen05-mma-scale-factor-b-block32-k96-ngt128-dig5.png](_images/tcgen05-mma-scale-factor-b-block32-k96-ngt128-dig5.png)


Figure 251 Layout of scale factor B matrix with block32 with K=96 and N>128 with SFA\_ID=11[](#tcgen05-mma-scale-factor-b-block32-k96-ngt128-dig5 "Permalink to this image")


![_images/tcgen05-mma-scale-factor-b-block32-k96-ngt128-dig6.png](_images/tcgen05-mma-scale-factor-b-block32-k96-ngt128-dig6.png)


Figure 252 Layout of scale factor B matrix with block32 with K=96 and N>128 with SFA\_ID=11[](#tcgen05-mma-scale-factor-b-block32-k96-ngt128-dig6 "Permalink to this image")

For example, if *SFB\_ID* is 0, then all the green columns are selected to form the
scale factor matrix. Similarly, *SFB\_ID* values of 1, 2 and 3 would select the blue,
yellow, and red columns, respectively.

###### 9.7.16.10.7.3.5. [Layout of the Scale Factor B Matrix for block16 with K=96 (Semantically equivalent to scale\_vec::6X)](#tcgen05-mma-scale-factor-b-layout-block16-k96)[](#tcgen05-mma-scale-factor-b-layout-block16-k96 "Permalink to this headline")

There are six scale factors per row of the `B` matrix with block size as 16 and the scale factor
must be provided in 4-byte aligned sub-column of the Tensor Memory. *SFB\_ID* specifies the byte
offset in the Tensor Memory word that must be used for the scale factor matrix.

For N<=128, [Figure 253](#tcgen05-mma-scale-factor-b-block16-k96-nlt128-dig1) and
[Figure 254](#tcgen05-mma-scale-factor-b-block16-k96-nlt128-dig2) show which sub-columns
get selected for different values of *SFB\_ID*.

![_images/tcgen05-mma-scale-factor-b-block16-k96-nlt128-dig1.png](_images/tcgen05-mma-scale-factor-b-block16-k96-nlt128-dig1.png)


Figure 253 Layout of scale factor B matrix with block16 with K=96 and N<=128 with SFA\_ID=00[](#tcgen05-mma-scale-factor-b-block16-k96-nlt128-dig1 "Permalink to this image")


![_images/tcgen05-mma-scale-factor-b-block16-k96-nlt128-dig2.png](_images/tcgen05-mma-scale-factor-b-block16-k96-nlt128-dig2.png)


Figure 254 Layout of scale factor B matrix with block16 with K=96 and N<=128 with SFA\_ID=10[](#tcgen05-mma-scale-factor-b-block16-k96-nlt128-dig2 "Permalink to this image")

For N>128, [Figure 255](#tcgen05-mma-scale-factor-b-block16-k96-ngt128-dig1),
[Figure 256](#tcgen05-mma-scale-factor-b-block16-k96-ngt128-dig2),
[Figure 257](#tcgen05-mma-scale-factor-b-block16-k96-ngt128-dig3) and
[Figure 258](#tcgen05-mma-scale-factor-b-block16-k96-ngt128-dig4) show which sub-columns
get selected for different values of *SFB\_ID*.

![_images/tcgen05-mma-scale-factor-b-block16-k96-ngt128-dig1.png](_images/tcgen05-mma-scale-factor-b-block16-k96-ngt128-dig1.png)


Figure 255 Layout of scale factor B matrix with block16 with K=96 and N>128 with SFA\_ID=00[](#tcgen05-mma-scale-factor-b-block16-k96-ngt128-dig1 "Permalink to this image")


![_images/tcgen05-mma-scale-factor-b-block16-k96-ngt128-dig2.png](_images/tcgen05-mma-scale-factor-b-block16-k96-ngt128-dig2.png)


Figure 256 Layout of scale factor B matrix with block16 with K=96 and N>128 with SFA\_ID=00[](#tcgen05-mma-scale-factor-b-block16-k96-ngt128-dig2 "Permalink to this image")


![_images/tcgen05-mma-scale-factor-b-block16-k96-ngt128-dig3.png](_images/tcgen05-mma-scale-factor-b-block16-k96-ngt128-dig3.png)


Figure 257 Layout of scale factor B matrix with block16 with K=96 and N>128 with SFA\_ID=10[](#tcgen05-mma-scale-factor-b-block16-k96-ngt128-dig3 "Permalink to this image")


![_images/tcgen05-mma-scale-factor-b-block16-k96-ngt128-dig4.png](_images/tcgen05-mma-scale-factor-b-block16-k96-ngt128-dig4.png)


Figure 258 Layout of scale factor B matrix with block16 with K=96 and N>128 with SFA\_ID=10[](#tcgen05-mma-scale-factor-b-block16-k96-ngt128-dig4 "Permalink to this image")

For example, if *SFB\_ID* is 0, then all the green columns are selected to form the
scale factor matrix. Similarly, if *SFB\_ID* is 2, then all of the blue columns are
selected to form the scale factor matrix.