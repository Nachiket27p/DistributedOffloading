
class WorkerList:
    """This class creates a dsta structure which is used to keep
    tracl of the workers which have connected to the main node.
    There are two possible state the worker can be in, either
    occupied or free. Depending on which state they are in, they
    will be placed in the appropriate set.
    """
    def __init__(self):
        """Construct the worker list with two sets.
        """
        self.__free = set()
        self.__occupied = set()

    def addWorker(self, worker):
        """Add a worker to the list, this will add the worker to the
        free set.

        Args:
            worker (tuple): A tuple with two values, the first is the Ip address
            the second is the port number.
        """
        self.__free.add(worker)

    def getWorker(self):
        """Returns a worker from the worker list if ther is one available.

        Raises:
            None: If a request for a worker is made when there are no free workers.

        Returns:
            tuple: A tuple containing two values, the first being the ip address and,
            the second being the port number.
        """
        if len(self.__free):
            w = self.__free.pop()
            self.__occupied.add(w)
            return w
        else:
            raise None

    def freeWorker(self, worker):
        """Used to replace a worker which was used back into the free set so it can be
        reused if required.

        Args:
            worker (tuple): A tuple with two values, the first is the Ip address
            the second is the port number.

        Raises:
            KeyError: If the worker provided in not present in the occuped list
            then it is one which is not recognized so raise a key error.
        """
        if worker in self.__occupied:
            self.__occupied.remove(worker)
            self.__free.add(worker)
        else:
            raise KeyError

    def freeSize(self):
        """Get the number of free workers in the free set.

        Returns:
            int: Number of free workers.
        """
        return len(self.__free)

    def occupiedSize(self):
        """Get the number of workers which are occupied.

        Returns:
            int: Number of worker in the occupied set.
        """
        return len(self.__occupied)

    def destroyWorkers(self):
        """Deletes all the workers from both the free and occupied set.
        """
        while(len(self.__free)):
            w = self.__free.pop()
            w.close()

        while(len(self.__occupied)):
            w = self.__occupied.pop()
            w.close()
