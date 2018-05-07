"""Local task-pull routine for bias recovery testing and profiling (one process only).

Note
----
Profile with cProfile and store as .prof to be visualized with https://jiffyclub.github.io/snakeviz/.
"""
from __future__ import division, absolute_import

import sys
import json
import abc

from bcn.examples import figures as parallel # NOTE If parallel needs to be changed, do it here too.

    
if __name__ == '__main__':
    
    class_ = sys.argv[1]
    kwargs = json.loads(sys.argv[2])

    taskpull = getattr(parallel, class_)(**kwargs)

    tasks = taskpull.create_tasks()
    taskpull.allocate()

    for task in tasks:
        result = taskpull.work(task)
        taskpull.store(result)

    taskpull.postprocessing()

    

