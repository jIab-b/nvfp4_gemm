### 8.7.2. Limitations on Mixed-size Data-races 

A *data-race* between operations that *overlap* completely is called a *uniform-size data-race*,
while a *data-race* between operations that *overlap* partially is called a *mixed-size data-race*.

The axioms in the memory consistency model do not apply if a PTX program contains one or more
*mixed-size data-races*. But these axioms are sufficient to describe the behavior of a PTX program
with only *uniform-size data-races*.

Atomicity of mixed-size RMW operations

In any program with or without *mixed-size data-races*, the following property holds for every pair
of *overlapping atomic* operations A1 and A2 such that each specifies a *scope* that includes the
other: Either the *read-modify-write* operation specified by A1 is performed completely before A2 is
initiated, or vice versa. This property holds irrespective of whether the two operations A1 and A2
overlap partially or completely.