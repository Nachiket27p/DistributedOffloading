# Matrix Multiplication Offloading in Distributed Systems

## Info
* There are 4 essential components to this system:
  1) distributedMM.py
  2) transportMM.py
  3) workerList.py
  4) workerNode.py
* On the main node side, numbers 1, 2, and 3 are required
* On the worker node side, numbers 2, and  4 are required

* This system can be used comletely on a local computer but there are some small modification which need to be made
  * First isto change the 'hip' an 'ip' address to local values instead of '10.142.0.2' and 'socket.gethostbyname(socket.gethostname())' respectively.
  * Another is to keep in mind the worker node computes its task utilizing all possible cores so the speed up benefits displayed in the paper will not be visible because all the workers will compete for all the cpu cores.
  * The times observed when run completely locally will be slower than just computing locally.

## Pre requirements
* The system is developed fully within python so only a limited number of packages are required
* The system was developed for python 3, more specifically (3.8 was used but versions greater than 3.6 should work fine)
* The required packages are listed in the 'requirements.txt' file provided
* Use the following command to install the required packages to python

        $ pip install --user requirements.txt

## How to use?
* All the test files used to collect the data are provided in the 'test' directory
* To any of the files please move them to the root directory where 'distributedMM.py', 'transportMM.py', 'workerList.py', and 'workerNode.py' are locate

### How to start the main node?
1) An example file by the name of 'matMult.py' is provided in the 'test' directory along with other test files which shows how to import the 'DMM' class from the 'distributedMM.py' file and condigure values.
2) Creating an instance of the 'DMM' class will launch a thread which will start listening for worker nodes trying to connect, I would suggest using an input statement to allow you to connect the worker nodes to the main node.
3) At this point refer to the first point in the "How to start worker node?" section
4) Now proceed past the input step in your code, and the matrix multiplication should be computed and you should be able to see the terminal output on the worker nodes show when they receive work.

### How to start worker node?
1) First ensure you have completed the first two steps of the "How to start the main node?" section before proceeding
2) Ensure the 'hip', 'ip' and 'port' values are correctly assigned, for example:

        If the Main node has ip = 127.0.0.1 and port = 5000
        Then ensure hip = 12.0.0.1 and port = 5000 in the 'workerNode.py' file before starting it
        Also ensure that the ip variable in 'workerNode.py' is the correct address of the worker node

3) Next to start the worker node run the following command, and a 'connected' message should be printed to the terminal as shown below

        $ python workerNode.py
        Connected

4) If the specific ip and port are not correct/available the following error message will be displayed indicating the error
5) Hopefully no error encountered and the 'connected' message was printed, then please look at step 4 in the "How to start the main node?"