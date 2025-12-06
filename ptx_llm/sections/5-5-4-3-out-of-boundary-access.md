#### 5.5.4.3. Out of Boundary Access 

In im2col mode, when the number of requested pixels in NDHW space specified by *Pixels-per-Column*
exceeds the number of available pixels in the image batch then out-of-bounds access is performed.

Similar to tiled mode, zero fill or `OOB-NaN` fill can be performed based on the Fill-Mode
specified.