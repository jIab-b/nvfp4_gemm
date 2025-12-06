###### 9.7.9.25.3.2. Data Movement and Conversion Instructions: cp.async.commit_group 

`cp.async.commit_group`

Commits all prior initiated but uncommitted `cp.async` instructions into a *cp.async-group*.

Syntax

```
cp.async.commit_group ;
```

Description

`cp.async.commit_group` instruction creates a new *cp.async-group* per thread and batches all
prior `cp.async` instructions initiated by the executing thread but not committed to any
*cp.async-group* into the new *cp.async-group*. If there are no uncommitted `cp.async`
instructions then `cp.async.commit_group` results in an empty *cp.async-group.*

An executing thread can wait for the completion of all `cp.async` operations in a *cp.async-group*
using `cp.async.wait_group`.

There is no memory ordering guarantee provided between any two `cp.async` operations within the
same *cp.async-group*. So two or more `cp.async` operations within a *cp.async-group* copying data
to the same location results in undefined behavior.

PTX ISA Notes

Introduced in PTX ISA version 7.0.

Target ISA Notes

Requires `sm_80` or higher.

Examples

```
// Example 1:

cp.async.ca.shared.global [shrd], [gbl], 4;

cp.async.commit_group ; // Marks the end of a cp.async group



// Example 2:

cp.async.ca.shared.global [shrd1],   [gbl1],   8;

cp.async.ca.shared.global [shrd1+8], [gbl1+8], 8;

cp.async.commit_group ; // Marks the end of cp.async group 1



cp.async.ca.shared.global [shrd2],    [gbl2],    16;

cp.async.cg.shared.global [shrd2+16], [gbl2+16], 16;

cp.async.commit_group ; // Marks the end of cp.async group 2
```