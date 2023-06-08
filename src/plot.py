from matplotlib import pyplot as plt
import numpy as np
from src.model_base import ModelBase
import os 
from pathlib import Path
from pactools import Comodulogram, REFERENCES
from pactools import simulate_pac
from scipy.fft import fft, fftfreq

class Plot:
    def __init__(self, addresses, *, t_start=None, t_end=None, title: str = None, file: Path = None , plot_type: str = "timeseries"):
        self.addresses = addresses
        self.t_start = t_start
        self.t_end = t_end
        self.title = title
        self.file = file
        self.plot_type = plot_type
        self.check_path()
        return

    def check_path(self):
        self.file.parent.mkdir(exist_ok=True)

    def __call__(self, t: np.array, res: list[ModelBase],**kwargs):
        if self.plot_type == "timeseries":
            self.plot(t, res,**kwargs)
        elif self.plot_type == "pac":
            self.plot_pac(t,res,**kwargs)
        elif self.plot_type == "fft":
            self.plot_fft(t,res,**kwargs)
        return

    def get_traces(self, t: np.array, res: list[ModelBase]):
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
        return (t[_t], traces)

    def plot(self, t: np.array, res: list[ModelBase]):
        t_t, traces = self.get_traces(t, res)
        for tr in traces:
            plt.plot(t_t, tr)
        if self.title is not None:
            plt.title(self.title)
        self.save()
        return

    def plot_pac(self, t: np.array, res: list[ModelBase]):
        t_t, traces = self.get_traces(t, res)
        s = traces[0]
        fs = 1/(t[1]-t[0])
        low_fq_range = np.linspace(1, 7, 40)
        methods = [
            'ozkurt', 'canolty', 'tort', 'penny', 'vanwijk', 'duprelatour', 'colgin',
            'sigl', 'bispectrum'
        ]
        low_fq_width = 1.0  # Hz

        n_lines = 3
        n_columns = int(np.ceil(len(methods) / float(n_lines)))

        fig, axs = plt.subplots(
            n_lines, n_columns, figsize=(4 * n_columns, 3 * n_lines))
        axs = axs.ravel()
        for ax, method in zip(axs, methods):
            #print('%s... ' % (method, ))
            estimator = Comodulogram(fs=fs, low_fq_range=low_fq_range,
                                     low_fq_width=low_fq_width, method=method,
                                     progress_bar=False)
            estimator.fit(s)
            estimator.plot(titles=[REFERENCES[method]], axs=[ax])
            ax.set_ylim([0, 100])

        self.save()
        return

    def plot_fft(self, t:np.array, res: list[ModelBase],min_fq=1, max_fq=50):
        t_t, traces = self.get_traces(t, res)
        s = traces[0]
        dt = t[1]-t[0]
        N = len(s)
        yf = fft(s)
        xf = fftfreq(N, dt)[:N//2]
        limit = np.where((xf<=max_fq)& (xf>=min_fq))
        yf = yf[0:N//2]
        plt.plot(xf[limit], 2.0/N * np.abs(yf[limit]))
        self.save()
        return
    
    def max_gamma_power(self, t:np.array, res: list[ModelBase],min_fq=10, max_fq=50):
        t_t, traces = self.get_traces(t, res)
        s = traces[0]
        dt = t[1]-t[0]
        N = len(s)
        yf = fft(s)
        xf = fftfreq(N, dt)[:N//2]
        limit = np.where((xf<=max_fq)& (xf>=min_fq))
        yf = yf[0:N//2]
        max_ind = xf.index(max(xf[limit]))
        Max_freq = max(xf[limit])
        max_freq_power = yf[max_ind]
        return Max_freq, max_freq_power
        # input between 0 and 1 and 10 values
        # average over 5 up to 20 trials
        # each trial throw away 5s and simulate for a 100s (because its gamma maybe you can do a bit shorter)
        # dt one order of magnitude smaller than taus so 0.1 ms
        
        
    def save(self):
        if self.file is None:
            plt.show()
            return
        path = Path(self.file)
        ext = path.suffix
        if ext.lower() == '.png':
            plt.savefig(self.file, format='png')
        elif ext.lower() == '.eps':
            plt.savefig(self.file, format='eps')
        else:
            plt.savefig(self.file)
        plt.close()
        return
