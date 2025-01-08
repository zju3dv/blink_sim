import time
# import torch
import numpy as np

class catchtime(object):
    def __enter__(self):
        # torch.cuda.synchronize()
        self.t = time.time()
        return self

    def __exit__(self, type, value, traceback):
        # torch.cuda.synchronize()
        self.t = time.time() - self.t