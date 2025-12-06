##### 9.7.18.2.2. SIMD Video Instructions: vset2 

`vset2`

Integer dual half-word SIMD comparison.

Syntax

```
// SIMD instruction with secondary SIMD merge operation

vset2.atype.btype.cmp  d{.mask}, a{.asel}, b{.bsel}, c;



// SIMD instruction with secondary accumulate operation

vset2.atype.btype.cmp.add  d{.mask}, a{.asel}, b{.bsel}, c;



.atype = .btype = { .u32, .s32 };

.cmp   = { .eq, .ne, .lt, .le, .gt, .ge };

.mask  = { .h0, .h1, .h10 };  // defaults to .h10

.asel  = .bsel  = { .hxy, where x,y are from { 0, 1, 2, 3 } };

   .asel defaults to .h10

   .bsel defaults to .h32
```

Description

Two-way SIMD parallel comparison with secondary operation.

Elements of each dual half-word source to the operation are selected from any of the four half-words
in the two source operands `a` and `b` using the `asel` and `bsel` modifiers.

The selected half-words are then compared in parallel.

The intermediate result of the comparison is always unsigned, and therefore the half-words of
destination `d` and operand `c` are also unsigned.

For instructions with a secondary SIMD merge operation:

* For half-word positions indicated in mask, the selected half-word results are copied into
  destination `d`. For all other positions, the corresponding half-word from source operand `b`
  is copied to `d`.

For instructions with a secondary accumulate operation:

* For half-word positions indicated in mask, the selected half-word results are added to operand
  `c`, producing `a` result in `d`.

Semantics

```
// extract pairs of half-words and sign- or zero-extend

// based on operand type

Va = extractAndSignExt_2( a, b, .asel, .atype );

Vb = extractAndSignExt_2( a, b, .bsel, .btype );

Vc = extractAndSignExt_2( c );

for (i=0; i<2; i++) {

    t[i] = compare( Va[i], Vb[i], .cmp ) ? 1 : 0;

}

// secondary accumulate or SIMD merge

mask = extractMaskBits( .mask );

if (.add) {

    d = c;

    for (i=0; i<2; i++) {  d += mask[i] ? t[i] : 0;  }

} else {

    d = 0;

    for (i=0; i<2; i++)  {  d |= mask[i] ? t[i] : Vc[i];  }

}
```

PTX ISA Notes

Introduced in PTX ISA version 3.0.

Target ISA Notes

`vset2` requires `sm_30` or higher.

Examples

```
vset2.s32.u32.lt      r1, r2, r3, r0;

vset2.u32.u32.ne.add  r1, r2, r3, r0;
```