#### 9.7.1.11. Integer Arithmetic Instructions: neg 

`neg`

Arithmetic negate.

Syntax

```
neg.type  d, a;



.type = { .s16, .s32, .s64 };
```

Description

Negate the sign of **a** and store the result in **d**.

Semantics

```
d = -a;
```

Notes

Only for signed integers.

PTX ISA Notes

Introduced in PTX ISA version 1.0.

Target ISA Notes

Supported on all target architectures.

Examples

```
neg.s32  r0,a;
```