#### 9.7.15.1. Warpgroup 

A warpgroup is a set of four contiguous warps such that the *warp-rank* of the first warp is a
multiple of 4.

warp-rank of a warp is defined as:

```
(%tid.x + %tid.y * %ntid.x  + %tid.z * %ntid.x * %ntid.y) / 32
```