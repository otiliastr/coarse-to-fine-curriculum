import numpy as np
import tensorflow as tf

__author__ = 'Otilia Stretcu'


def print_metrics_dict(metrics):
    for name, val in metrics.items():
        print('--------------', name, '--------------')
        if isinstance(val, tf.Tensor):
            val = val.numpy()
        if name == 'confusion':
            print(np.array2string(val, separator=', ', precision=2))
        else:
            print(val)
