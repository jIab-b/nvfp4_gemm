### 8.2.6. Initialization 

Each byte in memory is initialized by a hypothetical write *W0* executed before starting any thread
in the program. If the byte is included in a program variable, and that variable has an initial
value, then *W0* writes the corresponding initial value for that byte; else *W0* is assumed to have
written an unknown but constant value to the byte.