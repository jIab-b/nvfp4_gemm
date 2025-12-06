##### 9.7.16.8.2. Packing and Unpacking 

Optionally, the following pack and unpack operations can be performed during the load and store:

1. Packing: two 16-bit chunks can be packed into a single 32-bit chunk in the register in `tcgen05.ld`
2. Unpacking: a single 32-bit chunk in the register can be unpacked into two 16-bit chunks in `tcgen05.st`

as shown in the [Figure 193](#tcgen05-ld-st-pack-unpack).

![_images/tcgen05-ld-st-pack-unpack.png](_images/tcgen05-ld-st-pack-unpack.png)


Figure 193 Pack/Unpack operations for tcgen05 ld/st[](#tcgen05-ld-st-pack-unpack "Permalink to this image")