import atexit
from queue import Empty
# from threading import Thread
from multiprocessing import Queue, Process
from rllab.sampler.utils import rollout
import numpy as np
import pickle as pickle

__all__ = [
    'init_worker',
    'init_plot',
    'update_plot'
]

process = None
queue = None

def _worker_start():
    env = None
    policy = None
    max_length = None
    global queue, process
    try:
        while True:
            msgs = {}
            # Only fetch the last message of each type
            while True:
                try:
                    msg = queue.get_nowait()
                    msgs[msg[0]] = msg[1:]
                except Empty:
                    break
            if 'stop' in msgs:
                break
            elif 'update' in msgs:
                env, policy = msgs['update']
                # env.start_viewer()
            elif 'demo' in msgs:
                param_values, max_length = msgs['demo']
                policy.set_param_values(param_values)
                rollout(env, policy, max_path_length=max_length, animated=True, speedup=5)
            else:
                if max_length:
                    rollout(env, policy, max_path_length=max_length, animated=True, speedup=5)
    except KeyboardInterrupt:
        pass


def _shutdown_worker():
    if process or queue:
        queue.put(['stop'])
        process.join()


def init_worker():
    global process, queue
    if queue is None:
        queue = Queue()
        atexit.register(_shutdown_worker)
        process = Process(target=_worker_start)
        process.start()

def init_plot(env, policy, max_path_length):
    init_worker()
    queue.put(['update', env, policy])


def update_plot(policy, max_length=np.inf):
    queue.put(['demo', policy.get_param_values(), max_length])
