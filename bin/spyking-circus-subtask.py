#!/usr/bin/env python
'''
Script that launches a subtask. We cannot call functions directly from
the main spyking_circus script, since we want to start them with ``mpirun``.
'''
import sys
import circus


if __name__ == '__main__':
    # This should not never be called by the user, therefore we can assume a
    # standard format
    assert len(sys.argv) == 5, 'Incorrect number of arguments -- do not run this script manually, use "spyking-circus" instead'
    task = sys.argv[1]
    filename = sys.argv[2]
    nb_cpu = int(sys.argv[3])
    use_gpu = (sys.argv[4].lower() == 'true')
    circus.launch(task, filename, nb_cpu, use_gpu)