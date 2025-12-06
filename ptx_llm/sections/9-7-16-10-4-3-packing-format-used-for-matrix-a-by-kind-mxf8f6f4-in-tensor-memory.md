###### 9.7.16.10.4.3. Packing format used for matrix A by .kind::mxf8f6f4 in Tensor Memory 

The individual 4-bit and the 6-bit floating point type elements must be packed in an 8-bit container
in Tensor memory as shown below. The 8-bit containers must be contiguously packed in a 32-bit Tensor
Memory word. For example, if the type of elements of the matrix `A` is 6 bits then 4 consecutive
`A` elements should be packed in one 32-bit Tensor Memory word.

* 4-bit packing format as shown in [Figure 199](#tcgen05-packing-formats-mxf8f6f4-tmem-dig1)

  ![_images/tcgen05-packing-formats-mxf8f6f4-tmem-dig1.png](_images/tcgen05-packing-formats-mxf8f6f4-tmem-dig1.png)


  Figure 199 4-bit packing format with type E2M1[](#tcgen05-packing-formats-mxf8f6f4-tmem-dig1 "Permalink to this image")
* 6-bit packing format

  + Type E3M2 as shown in [Figure 200](#tcgen05-packing-formats-mxf8f6f4-tmem-dig2)

    ![_images/tcgen05-packing-formats-mxf8f6f4-tmem-dig2.png](_images/tcgen05-packing-formats-mxf8f6f4-tmem-dig2.png)


    Figure 200 6-bit packing format with type E3M2[](#tcgen05-packing-formats-mxf8f6f4-tmem-dig2 "Permalink to this image")
  + Type E2M3 as shown in [Figure 201](#tcgen05-packing-formats-mxf8f6f4-tmem-dig3)

    ![_images/tcgen05-packing-formats-mxf8f6f4-tmem-dig3.png](_images/tcgen05-packing-formats-mxf8f6f4-tmem-dig3.png)


    Figure 201 6-bit packing format with type E2M3[](#tcgen05-packing-formats-mxf8f6f4-tmem-dig3 "Permalink to this image")