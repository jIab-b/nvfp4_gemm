###### 9.7.16.10.7.3. Scale Factor B ID 

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