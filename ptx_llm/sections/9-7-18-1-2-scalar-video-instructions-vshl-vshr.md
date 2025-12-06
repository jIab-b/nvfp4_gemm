##### 9.7.18.1.2. Scalar Video Instructions: vshl , vshr 

`vshl`, `vshr`

Integer byte/half-word/word left/right shift.

Syntax

```
// 32-bit scalar operation, with optional secondary operation

vop.dtype.atype.u32{.sat}.mode       d, a{.asel}, b{.bsel};

vop.dtype.atype.u32{.sat}.mode.op2   d, a{.asel}, b{.bsel}, c;



// 32-bit scalar operation, with optional data merge

vop.dtype.atype.u32{.sat}.mode  d.dsel, a{.asel}, b{.bsel}, c;



 vop   = { vshl, vshr };

.dtype = .atype = { .u32, .s32 };

.mode  = { .clamp, .wrap };

.dsel  = .asel  = .bsel  = { .b0, .b1, .b2, .b3, .h0, .h1 };

.op2   = { .add, .min, .max };
```

Description

`vshl`
:   Shift `a` left by unsigned amount in `b` with optional saturate, and optional secondary
    arithmetic operation or subword data merge. Left shift fills with zero.

`vshr`
:   Shift `a` right by unsigned amount in `b` with optional saturate, and optional secondary
    arithmetic operation or subword data merge. Signed shift fills with the sign bit, unsigned shift
    fills with zero.

Semantics

```
// extract byte/half-word/word and sign- or zero-extend

// based on source operand type

ta = partSelectSignExtend( a,atype, asel );

tb = partSelectSignExtend( b, .u32, bsel );

if ( mode == .clamp  && tb > 32 )  tb = 32;

if ( mode == .wrap )                       tb = tb & 0x1f;

switch ( vop ){

   case vshl:  tmp = ta << tb;

   case vshr:  tmp = ta >> tb;

}

// saturate, taking into account destination type and merge operations

tmp = optSaturate( tmp, sat, isSigned(dtype), dsel );

d = optSecondaryOp( op2, tmp, c );  // optional secondary operation

d = optMerge( dsel, tmp, c );       // optional merge with c operand
```

PTX ISA Notes

Introduced in PTX ISA version 2.0.

Target ISA Notes

`vshl`, `vshr` require `sm_20` or higher.

Examples

```
vshl.s32.u32.u32.clamp  r1, r2, r3;

vshr.u32.u32.u32.wrap   r1, r2, r3.h1;
```