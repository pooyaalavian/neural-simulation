from matplotlib import pyplot as plt
import numpy as np
from src.model_base import ModelBase
import os 
from pathlib import Path


class Plot:
    def __init__(self, addresses, *, t_start=None, t_end=None, title: str = None, file: str = None):
        self.addresses = addresses
        self.t_start = t_start
        self.t_end = t_end
        self.title = title
        self.file = file
        self.check_path()
        return

    def check_path(self):
        dir_exists = os.path.isdir(os.path.dirname(self.file))
        if not dir_exists:
            os.makedirs(os.path.dirname(self.file), exist_ok=True)

    def __call__(self, t: np.array, res: list[ModelBase]):
        self.plot(t, res)
        return

    def plot(self, t: np.array, res: list[ModelBase]):
        traces = []
        if self.t_start is None:
            self.t_start = t[0]
        if self.t_end is None:
            self.t_end = t[-1]
        _t = np.where((t >= self.t_start) & (t <= self.t_end))
        idx_start = _t[0][0]
        idx_end = _t[0][-1]
        _res = res[idx_start:idx_end+1]
        for ad in self.addresses:
            tmp = [x for x in _res]
            for key in ad.split('.'):
                tmp = [getattr(x, key) for x in tmp]
            traces.append(np.array(tmp))
        for tr in traces:
            plt.plot(t[_t], tr)
        if self.title is not None:
            plt.title(self.title)
        self.save()
        return
    
    def save(self):
        if self.file is None:
            plt.show()
            return 
        path = Path(self.file)
        ext = path.suffix
        if ext.lower()=='.png':
            plt.savefig(self.file, format='png')
        elif ext.lower()=='.eps':
            plt.savefig(self.file, format='eps')
        else:
            plt.savefig(self.file)
        plt.close()
        return
