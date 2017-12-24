# Utilities

import time
import numpy as np

def timeit(f):
    def timed(*args, **kwargs):
        start_time = time.time()
        result = f(*args, **kwargs)
        end_time = time.time()

        print("   [-] %s : %2.5f sec" % (f.__name__, end_time - start_time))
        return result

    return timed

@timeit
def save_npy(obj, path):
    np.save(path, obj)
    print("  [*] save %s" % path)

@timeit
def load_npy(path):
    obj = np.load(path)
    print("  [*] load %s" % path)
    return obj