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
  * Presented more refined and narrowed project. ([Presentation])
    * Outlines a more precise implementation plan, with evaluation metrics.
* 03/26/21
  * Designed rudementary communication procedure for transmitting data between main node and worker nodes
  * Implemented logging mechanism to track each step on both main node and worker side
    * Logs should be present in the logs directory
    * Tracks:
      * sending and receiving of task info (header)
      * sending of matrix data
      * receiving of matrix data
      * sending of results
      * receivinf of results
  * Performed real world analysis on data size and time requirement for transporting data
    * The specific data collected is presented in the following spreadsheet: [Time_analysis_of_data_transfer]
  * The data analysis shows that raw encoded data transfer is too intensive on bandwidth and high transfer times
  * Utilized zfpy module to compress matrix data before sending the data
    * The compression is lossless, and ywilds ~10x compression
  * The time saved by compressing the data and sending it shows
    * Bandwidth usage is reduced by ~10x
    * Time to transfer data is reduced by ~6x
      * This is lower than the compression savings because of additional time to compress the data is incorporated into the sending and receiving time.
  
  * Need to further analyze the findings ...
  * 
* 03/29/21
  * Completed presentation for [mid-project] review
  * Analyzed previously connected data


...  


[proposal]:https://github.com/Nachiket27p/DistributedOffloading/blob/master/proposal/projectProposal.pdf
[Presentation]:https://github.com/Nachiket27p/DistributedOffloading/blob/master/proposal/presentation/presentation.pptx
<<<<<<< HEAD
[Time_analysis_of_data_transfer]:https://github.com/Nachiket27p/DistributedOffloading/blob/master/presentation/times.xlsx
[mid-project]:https://github.com/Nachiket27p/DistributedOffloading/blob/master/presentation/midProject.pptx
=======
[Time_analysis_of_data_transfer]:https://github.com/Nachiket27p/DistributedOffloading/blob/master/proposal/presentation/times.xlsx
>>>>>>> ff2be9bb4030487b9850a1415ccd1067ca768a83
