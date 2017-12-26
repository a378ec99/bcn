"""Task-pull routine for bias recovery.

Note
----
Task-Pull based on https://github.com/jbornschein/mpi4py-examples/blob/master/09-task-pull.py
"""
from __future__ import absolute_import


import sys # WARNING remove in final version
sys.path.append('/home/sohse/projects/PUBLICATION/GITssh/bcn')

#import sys
import json
import abc

from mpi4py import MPI
from bcn.examples import figures as parallel # DANGER


if __name__ == '__main__':
    
    class_ = sys.argv[1]
    kwargs = json.loads(sys.argv[2])
    READY, DONE, EXIT, START = range(4)
    
    comm = MPI.COMM_WORLD
    size = comm.size
    rank = comm.rank
    status = MPI.Status()
    node = MPI.Get_processor_name()
    
    try:
        master = comm.allgather(node).index('node01') #NOTE Hardcoded: This tries that post-processing and storage happens on the high-memory node (512GB).
        print master
    except ValueError:
        print 'Could not use node01 as master.'
        master = 0

    taskpull = getattr(parallel, class_)(**kwargs)
    
    if rank == master: #NOTE Master process executes code below.

        tasks = taskpull.create_tasks()
        taskpull.allocate()
        total_workers = size - 1
        closed_workers = 0

        while closed_workers < total_workers:
            result = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            source = status.Get_source()
            tag = status.Get_tag()

            if tag == READY:
                try:
                    task = tasks.next()
                    comm.send(task, dest=source, tag=START)
                except StopIteration:
                    comm.send(None, dest=source, tag=EXIT)
                except SystemError:
                    print 'Error: ', task, rank, source, tag

            elif tag == DONE:
                taskpull.store(result)

            elif tag == EXIT:
                closed_workers += 1

        taskpull.postprocessing()
    
    else: #NOTE Worker processes execute code below.

        while True:
            comm.send(None, dest=master, tag=READY)
            task = comm.recv(source=master, tag=MPI.ANY_TAG, status=status)
            tag = status.Get_tag()
            if tag == START:
                result = taskpull.work(task)
                comm.send(result, dest=master, tag=DONE)
            elif tag == EXIT:
                break

        comm.send(None, dest=master, tag=EXIT)
