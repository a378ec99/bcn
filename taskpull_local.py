"""Local task-pull routine for bias recovery testing and profiling (one process only).

Note
----
Profile with cProfile and store as .prof to be visualized with https://jiffyclub.github.io/snakeviz/.
"""
from __future__ import absolute_import

import sys
import json
import abc

import bcn


class TaskPull(object):
    '''
    Abstract class that denotes API to taskpull.py and taskpull_local.py.
    '''
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def allocate(self):
        pass

    @abc.abstractmethod
    def create_tasks(self):
        pass

    @abc.abstractmethod
    def work(self):
        pass

    @abc.abstractmethod
    def store(self):
        pass

    @abc.abstractmethod
    def postprocessing(self):
        pass

    
if __name__ == '__main__':
    
    class_ = sys.argv[1]
    kwargs = json.loads(sys.argv[2])

    taskpull = getattr(bcn, class_)(kwargs)

    tasks = taskpull.create_tasks()
    taskpull.allocate()

    for task in tasks:
        result = taskpull.work(task)
        taskpull.store(result)

    taskpull.postprocessing()

    

