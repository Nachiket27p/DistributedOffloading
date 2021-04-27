
class WorkerList:
    def __init__(self):
        self.__free = set()
        self.__occupied = set()

    def addWorker(self, worker):
        self.__free.add(worker)

    def getWorker(self):
        if len(self.__free):
            w = self.__free.pop()
            self.__occupied.add(w)
            return w
        else:
            raise None

    def freeWorker(self, worker):
        if worker in self.__occupied:
            self.__occupied.remove(worker)
            self.__free.add(worker)
        else:
            raise KeyError

    def freeSize(self):
        return len(self.__free)

    def occupiedSize(self):
        return len(self.__occupied)

    def destroyWorkers(self):
        while(len(self.__free)):
            w = self.__free.pop()
            w.close()

        while(len(self.__occupied)):
            w = self.__occupied.pop()
            w.close()
