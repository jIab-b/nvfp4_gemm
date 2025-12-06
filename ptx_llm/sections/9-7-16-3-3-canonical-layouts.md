##### 9.7.16.3.3. Canonical Layouts 

In terms of [CuTe layouts](https://docs.nvidia.com/cutlass/media/docs/cpp/cute/01_layout.html)
the canonical layout can be expressed as follows:

| Major- ness | Swizzling mode | Canonical Layout without swizzling | [Swizzling](https://github.com/NVIDIA/cutlass/blob/bf9da7b76c766d7ee7d536afc77880a4ef1f1156/include/cute/swizzle.hpp) on the previous column |
| --- | --- | --- | --- |
| MN- major | No-swizzling or Interleaved | ((T,1,m),(8,k)):((1,T,SBO),(1T,LBO)) | Swizzle<0, 4, 3> |
| 32B Swizzling | ((T,2,m),(8,k)):((1,T,LBO),(2T,SBO)) | Swizzle<1, 4, 3> |
| 64B Swizzling | ((T,4,m),(8,k)):((1,T,LBO),(4T,SBO)) | Swizzle<2, 4, 3> |
| 128B Swizzling | ((T,8,m),(8,k)):((1,T,LBO),(8T,SBO)) | Swizzle<3, 4, 3> |
| K- major | No-swizzling or Interleaved | ((8,m),(T,2k)):((1T,SBO),(1,LBO)) | Swizzle<0, 4, 3> |
| 32B Swizzling | ((8,m),(T,2k)):((2T,SBO),(1,T)) | Swizzle<1, 4, 3> |
| 64B Swizzling | ((8,m),(T,2k)):((4T,SBO),(1,T)) | Swizzle<2, 4, 3> |
| 128B Swizzling | ((8,m),(T,2k)):((8T,SBO),(1,T)) | Swizzle<3, 4, 3> |

where

* T = 128 / sizeof-elements-in-bits
  T represents scale factor which normalizes matrix element types to 128-bits.
* m represents the number of repeating patterns across rows.
* k represents the number of repeating patterns across columns.

Examples

* K-Major, no-swizzling and tf32 type: [Figure 188](#tcgen05-k-no-swizzle-tf32)

  ![_images/async-warpgroup-k-no-swizzle-tf32.png](_images/async-warpgroup-k-no-swizzle-tf32.png)


  Figure 188 K major, no-swizzling and tf32 type[](#tcgen05-k-no-swizzle-tf32 "Permalink to this image")

  the strides and related details are as follows:

  Exact layout : Swizzle<0,4,3> o ((8,2),(4,4)):((4,32),(1,64))

  Canonical Layout :Swizzle<0,4,3> o ((8,m),(T,2k)):((1T,SBO),(1,LBO))

  | Parameters | Value |
  | --- | --- |
  | T | 4 |
  | m | 2 |
  | k | 2 |
  | LBO (relative offset) | 64\*sizeof(tf32) |
  | SBO | 32\*sizeof(tf32) |
  | Encoding of LBO in descriptor | (LBO) >> 4 = 16 |
  | Encoding of SBO in descriptor | (SBO) >> 4 = 8 |
* K-Major, 32B swizzling and tf32 type: [Figure 189](#tcgen05-k-32b-swizzle-tf32)

  ![_images/async-warpgroup-k-32B-swizzle-tf32.png](_images/async-warpgroup-k-32B-swizzle-tf32.png)


  Figure 189 K major, 32B swizzling and tf32 type[](#tcgen05-k-32b-swizzle-tf32 "Permalink to this image")

  the strides and related details are as follows:

  Exact layout : Swizzle<1,4,3> o ((8,2),(4,4)):((8,64),(1,4))

  Canonical Layout :Swizzle<1,4,3> o ((8,m),(T,2k)):((2T,SBO),(1,T))

  | Parameters | Value |
  | --- | --- |
  | T | 4 |
  | m | 2 |
  | k | 2 |
  | LBO (relative offset) | NA |
  | SBO | 64\*sizeof(tf32) |
  | Encoding of LBO in descriptor | 1 (assumed) |
  | Encoding of SBO in descriptor | (SBO) >> 4 = 16 |
* MN-Major, no-swizzling and bf16 type: [Figure 190](#tcgen05-mn-no-swizzle-bf16)

  ![_images/async-warpgroup-mn-no-swizzle-bf16.png](_images/async-warpgroup-mn-no-swizzle-bf16.png)


  Figure 190 MN major, no-swizzling and bf16 type[](#tcgen05-mn-no-swizzle-bf16 "Permalink to this image")

  the strides and related details are as follows:

  Exact layout : Swizzle<0,4,3> o ((8,1,2),(8,2)):((1,8,64),(8,128))

  Canonical Layout :Swizzle<0,4,3> o ((T,1,m),(8,k)):((1,T,SBO),(1T,LBO))

  | Parameters | Value |
  | --- | --- |
  | T | 8 |
  | m | 2 |
  | k | 2 |
  | LBO (relative offset) | 128\*sizeof(bf16) |
  | SBO | 64\*sizeof(bf16) |
  | Encoding of LBO in descriptor | (LBO) >> 4 = 16 |
  | Encoding of SBO in descriptor | (SBO) >> 4 = 8 |
* MN-Major, 32B swizzling and bf16 type: [Figure 191](#tcgen05-mn-32b-swizzle-bf16)

  ![_images/async-warpgroup-mn-32B-swizzle-bf16.png](_images/async-warpgroup-mn-32B-swizzle-bf16.png)


  Figure 191 MN major, 32B swizzling and bf16 type[](#tcgen05-mn-32b-swizzle-bf16 "Permalink to this image")

  the strides and related details are as follows:

  Exact layout : Swizzle<1,4,3> o ((8,2,2),(8,2)):((1,8,128),(16,256))

  Canonical Layout :Swizzle<1,4,3> o ((T,2,m),(8,k)):((1,T,LBO),(2T,SBO))

  | Parameters | Value |
  | --- | --- |
  | T | 8 |
  | m | 2 |
  | k | 2 |
  | LBO (relative offset) | 128\*sizeof(bf16) |
  | SBO | 256\*sizeof(bf16) |
  | Encoding of LBO in descriptor | (LBO) >> 4 = 16 |
  | Encoding of SBO in descriptor | (SBO) >> 4 = 32 |
* MN-Major, 64B swizzling and bf16 type: [Figure 192](#tcgen05-mn-64b-swizzle-bf16)

  ![_images/async-warpgroup-mn-64B-swizzle-bf16.png](_images/async-warpgroup-mn-64B-swizzle-bf16.png)


  Figure 192 MN major, 64B swizzling and bf16 type[](#tcgen05-mn-64b-swizzle-bf16 "Permalink to this image")

  the strides and related details are as follows:

  Exact layout : Swizzle<2,4,3> o ((8,4,2),(8,2)):((1,8,256),(32,512))

  Canonical Layout :Swizzle<2,4,3> o ((T,4,m),(8,k)):((1,T,LBO),(4T,SBO))

  | Parameters | Value |
  | --- | --- |
  | T | 8 |
  | m | 2 |
  | k | 2 |
  | LBO (relative offset) | 256\*sizeof(bf16) |
  | SBO | 512\*sizeof(bf16) |
  | Encoding of LBO in descriptor | (LBO) >> 4 = 32 |
  | Encoding of SBO in descriptor | (SBO) >> 4 = 64 |