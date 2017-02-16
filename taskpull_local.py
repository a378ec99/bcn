import sys
import json

import bcn as module



if __name__ == '__main__':
    '''
    Task-Pull for one (local) process to do testing.
    '''
    class_ = sys.argv[1]
    kwargs = json.loads(sys.argv[2])

    taskpull = getattr(module, class_)(kwargs)

    tasks = taskpull.create_tasks()
    taskpull.allocate()

    for task in tasks:
        result = taskpull.work(task)
        taskpull.store(result)

    taskpull.postprocessing()
    
    
