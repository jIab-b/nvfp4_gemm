## 2.2. Thread Hierarchy 

The batch of threads that executes a kernel is organized as a grid. A grid consists of either
cooperative thread arrays or clusters of cooperative thread arrays as described in this section and
illustrated in [Figure 1](#grid-of-clusters-grid-with-ctas) and
[Figure 2](#grid-of-clusters-grid-with-clusters). *Cooperative thread arrays (CTAs)* implement CUDA
thread blocks and clusters implement CUDA thread block clusters.

### 2.2.1. [Cooperative Thread Arrays](#cooperative-thread-arrays)[](#cooperative-thread-arrays "Permalink to this headline")

The *Parallel Thread Execution (PTX)* programming model is explicitly parallel: a PTX program
specifies the execution of a given thread of a parallel thread array. A *cooperative thread array*,
or CTA, is an array of threads that execute a kernel concurrently or in parallel.

Threads within a CTA can communicate with each other. To coordinate the communication of the threads
within the CTA, one can specify synchronization points where threads wait until all threads in the
CTA have arrived.

Each thread has a unique thread identifier within the CTA. Programs use a data parallel
decomposition to partition inputs, work, and results across the threads of the CTA. Each CTA thread
uses its thread identifier to determine its assigned role, assign specific input and output
positions, compute addresses, and select work to perform. The thread identifier is a three-element
vector `tid`, (with elements `tid.x`, `tid.y`, and `tid.z`) that specifies the thread’s
position within a 1D, 2D, or 3D CTA. Each thread identifier component ranges from zero up to the
number of thread ids in that CTA dimension.

Each CTA has a 1D, 2D, or 3D shape specified by a three-element vector `ntid` (with elements
`ntid.x`, `ntid.y`, and `ntid.z`). The vector `ntid` specifies the number of threads in each
CTA dimension.

Threads within a CTA execute in SIMT (single-instruction, multiple-thread) fashion in groups called
*warps*. A *warp* is a maximal subset of threads from a single CTA, such that the threads execute
the same instructions at the same time. Threads within a warp are sequentially numbered. The warp
size is a machine-dependent constant. Typically, a warp has 32 threads. Some applications may be
able to maximize performance with knowledge of the warp size, so PTX includes a run-time immediate
constant, `WARP_SZ`, which may be used in any instruction where an immediate operand is allowed.

### 2.2.2. [Cluster of Cooperative Thread Arrays](#cluster-of-cooperative-thread-arrays)[](#cluster-of-cooperative-thread-arrays "Permalink to this headline")

Cluster is a group of CTAs that run concurrently or in parallel and can synchronize and communicate
with each other via shared memory. The executing CTA has to make sure that the shared memory of the
peer CTA exists before communicating with it via shared memory and the peer CTA hasn’t exited before
completing the shared memory operation.

Threads within the different CTAs in a cluster can synchronize and communicate with each other via
shared memory. Cluster-wide barriers can be used to synchronize all the threads within the
cluster. Each CTA in a cluster has a unique CTA identifier within its cluster
(*cluster\_ctaid*). Each cluster of CTAs has 1D, 2D or 3D shape specified by the parameter
*cluster\_nctaid*. Each CTA in the cluster also has a unique CTA identifier (*cluster\_ctarank*)
across all dimensions. The total number of CTAs across all the dimensions in the cluster is
specified by *cluster\_nctarank*. Threads may read and use these values through predefined, read-only
special registers `%cluster_ctaid`, `%cluster_nctaid`, `%cluster_ctarank`,
`%cluster_nctarank`.

Cluster level is applicable only on target architecture `sm_90` or higher. Specifying cluster
level during launch time is optional. If the user specifies the cluster dimensions at launch time
then it will be treated as explicit cluster launch, otherwise it will be treated as implicit cluster
launch with default dimension 1x1x1. PTX provides read-only special register
`%is_explicit_cluster` to differentiate between explicit and implicit cluster launch.

### 2.2.3. [Grid of Clusters](#grid-of-clusters)[](#grid-of-clusters "Permalink to this headline")

There is a maximum number of threads that a CTA can contain and a maximum number of CTAs that a
cluster can contain. However, clusters with CTAs that execute the same kernel can be batched
together into a grid of clusters, so that the total number of threads that can be launched in a
single kernel invocation is very large. This comes at the expense of reduced thread communication
and synchronization, because threads in different clusters cannot communicate and synchronize with
each other.

Each cluster has a unique cluster identifier (*clusterid*) within a grid of clusters. Each grid of
clusters has a 1D, 2D , or 3D shape specified by the parameter *nclusterid*. Each grid also has a
unique temporal grid identifier (*gridid*). Threads may read and use these values through
predefined, read-only special registers `%tid`, `%ntid`, `%clusterid`, `%nclusterid`, and
`%gridid`.

Each CTA has a unique identifier (*ctaid*) within a grid. Each grid of CTAs has 1D, 2D, or 3D shape
specified by the parameter *nctaid*. Thread may use and read these values through predefined,
read-only special registers `%ctaid` and `%nctaid`.

Each kernel is executed as a batch of threads organized as a grid of clusters consisting of CTAs
where cluster is optional level and is applicable only for target architectures `sm_90` and
higher. [Figure 1](#grid-of-clusters-grid-with-ctas) shows a grid consisting of CTAs and
[Figure 2](#grid-of-clusters-grid-with-clusters) shows a grid consisting of clusters.

Grids may be launched with dependencies between one another - a grid may be a dependent grid and/or
a prerequisite grid. To understand how grid dependencies may be defined, refer to the section on
*CUDA Graphs* in the *Cuda Programming Guide*.

![Grid with CTAs](_images/grid-with-CTAs.png)


Figure 1 Grid with CTAs[](#grid-of-clusters-grid-with-ctas "Permalink to this image")


![Grid with clusters](_images/grid-with-clusters.png)


Figure 2 Grid with clusters[](#grid-of-clusters-grid-with-clusters "Permalink to this image")

A cluster is a set of cooperative thread arrays (CTAs) where a CTA is a set of concurrent threads
that execute the same kernel program. A grid is a set of clusters consisting of CTAs that
execute independently.