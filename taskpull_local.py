import sys
import json

import bcn



if __name__ == '__main__':
    '''
    Task-Pull for one (local) process to do testing and profiling.
    '''
    class_ = sys.argv[1]
    kwargs = json.loads(sys.argv[2])

    taskpull = getattr(bcn, class_)(kwargs)

    tasks = taskpull.create_tasks()
    taskpull.allocate()

    for task in tasks:
        result = taskpull.work(task)
        taskpull.store(result)

    taskpull.postprocessing()
    
    
# TODO Profile with cProfile again and store as .prof to be visualized with https://jiffyclub.github.io/snakeviz/. 
