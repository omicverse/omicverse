import sys, os
comm = None
from collections import namedtuple
import numpy as np


def mpi_init():
    if sys.platform.startswith('win32'):
        print("mpi4py does not work properly on Windows, so its use will be skipped.")
        return
    try:
        # test = 1 / 0
        from mpi4py import MPI
    except ImportError:
    # except ZeroDivisionError:
        print("Since the mpi4py-package is not installed, parallel computation is not possible. "
              "This package does not work properly on Windows, but can be installed on other operating systems.")
        return

    global comm
    global MPI
    comm = MPI.COMM_WORLD
    os.environ['OPENBLAS_NUM_THREADS'] = '1'


def get_mpi_info(singleProcess=False):
    MpiInfo = namedtuple("MpiInfo", ['rank', 'size'])
    if not singleProcess:
        return MpiInfo(get_process_rank(), get_process_size())
    else:
        return MpiInfo(0, 1)


def get_process_rank():
    return comm.Get_rank() if comm is not None else 0


def get_process_size():
    return comm.Get_size() if comm is not None else 1


def is_first_process():
    return get_process_rank() == 0


def world_allgather(data):
    return comm.allgather(data) if comm is not None else [data]


def gather(data, root=0):
    return comm.gather(data, root) if comm is not None else [data]


def Gather(data, root=0):
    return comm.Gather(data, root=root) if comm is not None else [data]


def GatherNpUnknownSize(data, root=0):
    """
    Gathers numpy arrays with unknown number of rows and set number of columns from all processes on root-process.
    :param data:
    :param root:
    :return:
    """
    n_rows, n_cols = data.shape
    mpi_info = get_mpi_info()

    # Compute sizes and offsets for Allgatherv
    sizes = np.array(comm.gather(n_rows, root=root))
    
    if mpi_info.rank == 0:
        sizes_memory = (n_cols * sizes).tolist()
        offsets = np.zeros(get_process_size(), dtype=int)
        offsets[1:] = np.cumsum(sizes_memory)[:-1].tolist()

        # Prepare buffer for Gatherv
        data_out = np.empty((np.sum(sizes), n_cols), dtype=np.double)
    else:
        data_out = None
        sizes_memory = None
        offsets = None
    comm.Gatherv(data, recvbuf=[data_out, sizes_memory, offsets, MPI.DOUBLE], root=root)

    return data_out


def Bcast(data, root=0, type='double'):
    if type == 'double':
        return comm.Bcast([data, MPI.DOUBLE], root=root) if comm is not None else None
    elif type == 'int':
        return comm.Bcast([data, MPI.INT], root=root) if comm is not None else None
    return comm.Bcast(data, root) if comm is not None else data


def bcast(data, root=0):
    return comm.bcast(data, root) if comm is not None else data


def barrier():
    return comm.barrier() if comm else None
