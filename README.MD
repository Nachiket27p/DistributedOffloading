# Task Offloading in Distributed Systems

## Project Abstract
* Heterogeneous distributed systems are more involved when it comes to balancing work loads due to the varying performance level of machines in the network.
* With an increase in the number of mobile devices and more reliable internet connections offloading demanding workloads to more powerful machines in a network can provide performance improvement.
* Tasks can be dynamically or statically offloaded, either whole tasks can be offloaded or parts of a task can be offloaded.
* Several things need to be considered before offloading a task; determine if the task can be offloaded effectively, which machines on the network can accept offloaded tasks, if the overhead of offloading is less than the overhead of waiting to run on the same machines.
* The full [proposal].

## Narrowed the scope
* The motivation is still the same but with a more focused area. The are in question is matrix multiplication, with the increasing popolarity of AI and Machine learning for systems without GPU's a large amount of CPU time is occupies by matrix multiplication.
* Some of the objetives are:

  1. To detect potential node failures and find another host to distribute the work load to.
  2. Efficiently transport the matrix data between machines.
  3. Aggregate the results and and produce the final output.
  4. Explore different levels of granularity to find the optimal number of partitions.

* Some of the performance evaluation metrics.
  1. The speedup compared to performing computation on single machines.
  2. The optimal granularity of task offloading.
  3. The maximum number of node failures before the overhead of re-offloading diminishes any performance gains.

* 

## Progress Timeline
* 02/15/21
  * General project [proposal].
* 02/28/21
  * Researched more specialized topics involving task offloading within distributed systems.
* 03/04/21
  * Narrowed down scope to distributed matrix multipication.
* 03/09/21
  * Presented more refined an narrowed project. ([Presentation])
    * Outlines a more precise implementation plan, with evaluation metrics.



...  


## Implementation
* 

...

[proposal]:https://github.com/Nachiket27p/DistributedOffloading/blob/master/proposal/projectProposal.pdf
[Presentation]:https://github.com/Nachiket27p/DistributedOffloading/blob/master/proposal/presentation/presentation.pptx
