## 1.3. PTX ISA Version 9.0 

PTX ISA version 9.0 introduces the following new features:

* Adds support for `sm_88` target architecture.
* Adds support for `sm_110` target architecture.
* Adds support for target `sm_110f` that supports family-specific features.
* Adds support for target `sm_110a` that supports architecture-specific features.
* Adds support for pragma `enable_smem_spilling` that is used to enable shared
  memory spilling for a function.
* Adds support for pragma `frequency` that is used to specify the execution frequency of a basic
  block.
* Adds support for directive `.blocksareclusters` that is used to specify that CUDA thread blocks
  are mapped to clusters.
* Extends `size` operand of `st.bulk` instruction to support 32-bit length.
* Adds support for performance-tuning directives `.abi_preserve` and `.abi_preserve_control`
  that are used to specify the number of data and control registers that should be preserved by the
  callers of a function.