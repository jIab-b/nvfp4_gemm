## 7.3. Alloca 

PTX provides `alloca` instruction for allocating storage at runtime on the per-thread local memory
stack. The allocated stack memory can be accessed with `ld.local` and `st.local` instructions
using the pointer returned by `alloca`.

In order to facilitate deallocation of memory allocated with `alloca`, PTX provides two additional
instructions: `stacksave` which allows reading the value of stack pointer in a local variable, and
`stackrestore` which can restore the stack pointer with the saved value.

`alloca`, `stacksave`, and `stackrestore` instructions are described in
[Stack Manipulation Instructions](#stack-manipulation-instructions).

Preview Feature

Stack manipulation instructions `alloca`, `stacksave` and `stackrestore` are preview features
in PTX ISA version 7.3. All details are subject to change with no guarantees of backward
compatibility on future PTX ISA versions or SM architectures.