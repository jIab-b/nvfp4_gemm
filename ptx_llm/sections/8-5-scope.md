## 8.5. Scope 

Each *strong* operation must specify a *scope*, which is the set of threads that may interact
directly with that operation and establish any of the relations described in the memory consistency
model. There are four scopes:

Table 21 Scopes[](#id674 "Permalink to this table")




| Scope | Description |
| --- | --- |
| `.cta` | The set of all threads executing in the same CTA as the current thread. |
| `.cluster` | The set of all threads executing in the same cluster as the current thread. |
| `.gpu` | The set of all threads in the current program executing on the same compute device as the current thread. This also includes other kernel grids invoked by the host program on the same compute device. |
| `.sys` | The set of all threads in the current program, including all kernel grids invoked by the host program on all compute devices, and all threads constituting the host program itself. |

Note that the warp is not a *scope*; the CTA is the smallest collection of threads that qualifies as
a *scope* in the memory consistency model.