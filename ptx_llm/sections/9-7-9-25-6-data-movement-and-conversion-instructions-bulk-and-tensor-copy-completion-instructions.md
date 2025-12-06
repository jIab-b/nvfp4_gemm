##### 9.7.9.25.6. Data Movement and Conversion Instructions: Bulk and Tensor copy completion instructions 

###### 9.7.9.25.6.1. [Data Movement and Conversion Instructions: `cp.async.bulk.commit_group`](#data-movement-and-conversion-instructions-cp-async-bulk-commit-group)[](#data-movement-and-conversion-instructions-cp-async-bulk-commit-group "Permalink to this headline")

`cp.async.bulk.commit_group`

Commits all prior initiated but uncommitted `cp.async.bulk` instructions into a
*cp.async.bulk-group*.

Syntax

```
cp.async.bulk.commit_group;
```

Description

`cp.async.bulk.commit_group` instruction creates a new per-thread *bulk async-group* and batches
all prior `cp{.reduce}.async.bulk.{.prefetch}{.tensor}` instructions satisfying the following
conditions into the new *bulk async-group*:

* The prior `cp{.reduce}.async.bulk.{.prefetch}{.tensor}` instructions use *bulk\_group* based
  completion mechanism, and
* They are initiated by the executing thread but not committed to any *bulk async-group*.

If there are no uncommitted `cp{.reduce}.async.bulk.{.prefetch}{.tensor}` instructions then
`cp.async.bulk.commit_group` results in an empty *bulk async-group*.

An executing thread can wait for the completion of all
`cp{.reduce}.async.bulk.{.prefetch}{.tensor}` operations in a *bulk async-group* using
`cp.async.bulk.wait_group`.

There is no memory ordering guarantee provided between any two
`cp{.reduce}.async.bulk.{.prefetch}{.tensor}` operations within the same *bulk async-group*.

PTX ISA Notes

Introduced in PTX ISA version 8.0.

Target ISA Notes

Requires `sm_90` or higher.

Examples

```
cp.async.bulk.commit_group;
```

###### 9.7.9.25.6.2. [Data Movement and Conversion Instructions: `cp.async.bulk.wait_group`](#data-movement-and-conversion-instructions-cp-async-bulk-wait-group)[](#data-movement-and-conversion-instructions-cp-async-bulk-wait-group "Permalink to this headline")

`cp.async.bulk.wait_group`

Wait for completion of *bulk async-groups*.

Syntax

```
cp.async.bulk.wait_group{.read} N;
```

Description

`cp.async.bulk.wait_group` instruction will cause the executing thread to wait until only N or
fewer of the most recent *bulk async-groups* are pending and all the prior *bulk async-groups*
committed by the executing threads are complete. For example, when N is 0, the executing thread
waits on all the prior *bulk async-groups* to complete. Operand N is an integer constant.

By default, `cp.async.bulk.wait_group` instruction will cause the executing thread to wait until
completion of all the bulk async operations in the specified *bulk async-group*. A bulk async
operation includes the following:

* Optionally, reading from the tensormap.
* Reading from the source locations.
* Writing to their respective destination locations.
* Writes being made visible to the executing thread.

The optional `.read` modifier indicates that the waiting has to be done until all the bulk
async operations in the specified *bulk async-group* have completed:

1. reading from the tensormap
2. the reading from their source locations.

PTX ISA Notes

Introduced in PTX ISA version 8.0.

Target ISA Notes

Requires `sm_90` or higher.

Examples

```
cp.async.bulk.wait_group.read   0;

cp.async.bulk.wait_group        2;
```