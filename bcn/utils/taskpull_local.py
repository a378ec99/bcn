"""Local task-pull (single process).
"""
from __future__ import division, absolute_import

import sys
import json
import abc

# NOTE If parallel needs to be changed, do it here too.
from bcn.examples import figures as parallel


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
