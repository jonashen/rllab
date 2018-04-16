import atexit
from queue import Empty, Queue
from threading import Thread
from rllab.sampler.utils import rollout
import numpy as np
import tensorflow as tf
import pickle as pickle

__all__ = [
    'init_worker',
    'init_plot',
    'update_plot'
]

process = None
queue = None
sess = None

def _worker_start():
    env = None
    policy = None
    max_length = None
    global queue, process, sess
    try:
        with sess.as_default():
            with sess.graph.as_default():
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
                        if not sess._closed:
                            rollout(env, policy, max_path_length=max_length, animated=True, speedup=5)
                    else:
                        if max_length:
                            if not sess._closed:
                                rollout(env, policy, max_path_length=max_length, animated=True, speedup=5)
    except KeyboardInterrupt:
        pass


def _shutdown_worker():
    if process or queue:
        queue.put(['stop'])
        queue.task_done()
        queue.join()
        process.join()


def init_worker():
    global process, queue, sess
    queue = Queue()
    sess = tf.get_default_session()
    process = Thread(target=_worker_start)
    process.daemon = True
    process.start()
    atexit.register(_shutdown_worker)


def init_plot(env, policy, max_path_length):
    # Call rollout once to display the window
    rollout(env, policy, animated=True, max_path_length=max_path_length)
    
    if queue is not None:
        _shutdown_worker().join()
    init_worker()
    queue.put(['update', env, policy])
    queue.task_done()


def update_plot(policy, max_length=np.inf):
    queue.put(['demo', policy.get_param_values(), max_length])
    queue.task_done()