from typing import Union
import numpy as np
from src.mixin import Mixin

zero = np.float64(0.0)


class NodeParamIback (Mixin):
    Keys = ['type', 'amplitude', 'frequency', 'phase', 'dc']

    def __init__(self, d: Union[dict, float], *, delta=False):
        self.__delta = delta
        dc_val = None
        if d is None:
            d = {}
        if type(d) in [float, int]:
            dc_val = d
            d = {}
        self.type = d.get('type', None)
        self.amplitude = self.__npget__(d, 'amplitude', delta=self.__delta)
        self.frequency = self.__npget__(d, 'frequency', delta=self.__delta)
        self.phase = self.__npget__(d, 'phase', delta=self.__delta)
        self.dc = self.__npget__(d, 'dc', delta=self.__delta)
        if dc_val is not None:
            self.dc = dc_val

        return

    def value(self, t) -> np.float64:
        if self.type == "sin":
            return self.amplitude * np.sin(2 * np.pi * self.frequency * t + self.phase) + self.dc
        elif self.type == "dc":
            return self.dc
        else:
            return self.dc

    def __flat_json__(self, *, ignore_zeros=False):
        if ignore_zeros:
            if self.type in ['dc', None] and self.dc == 0:
                return {}
            if self.type == 'sin' and self.amplitude == 0 and self.dc == 0:
                return {}
        val = ''
        if self.type in ['dc', None]:
            val = f'{self.dc}'
        elif self.type == 'sin':
            val = f'{self.amplitude} * sin(2 pi {self.frequency} * t + {self.phase}) + {self.dc}'
        return {"value": val}


class NodeParamIext(Mixin):
    Keys = ['type', 'height', 't_start', 't_end']

    def __init__(self, d: Union[dict, float], *, delta=False):
        self.__delta = delta
        dc_val = None
        if d is None:
            d = {}
        if type(d) in [float, int]:
            dc_val = d
            d = {}
        self.type = d.get('type', None)
        self.height = self.__npget__(d, 'height', self.__delta)
        self.t_start = self.__npget__(d, 't_start', self.__delta)
        self.t_end = self.__npget__(d, 't_end', self.__delta)
        if dc_val is not None:
            self.height = dc_val

    def value(self, t) -> np.float64:
        if self.type == "rectangular":
            if t >= self.t_start and t <= self.t_end:
                return self.height
            return zero
        
        if self.type == "dc":
            return self.height
        return zero


class NodeParam(Mixin):
    Keys = ['tau', 'sigma', 'gamma','gamma_r', 'tau_ampa',
            'sigma_ampa', 'gamma_ampa', 'I_back', 'I_ext']

    def __init__(self, d: dict, *, delta=False):
        self.__delta = delta
        if d is None:
            d = {}
        self.tau = self.__npget__(d, 'tau', delta=self.__delta)
        self.sigma = self.__npget__(d, 'sigma', delta=self.__delta)
        self.gamma = self.__npget__(d, 'gamma', delta=self.__delta)
        self.gamma_r = self.__npget__(d, 'gamma_r', delta=self.__delta)
        self.tau_ampa = self.__npget__(d, 'tau_ampa', delta=self.__delta)
        self.sigma_ampa = self.__npget__(d, 'sigma_ampa', delta=self.__delta)
        self.gamma_ampa = self.__npget__(d, 'gamma_ampa', delta=self.__delta)
        self.I_back = NodeParamIback(d.get('I_back', None), delta=self.__delta)
        self.I_ext = NodeParamIext(d.get('I_ext', None), delta=self.__delta)
        return
