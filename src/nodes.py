import json
import numpy as np

zero = np.float64(0.0)


class NodeParamIback:
    Keys = ['type', 'amplitude', 'frequency', 'phase', 'dc']

    def __init__(self, d: dict | float):
        dc_val = 0.0
        if d is None:
            d = {}
        if type(d) in [float, int]:
            dc_val = d
            d = {}
        self.type = d.get('type', None)
        self.amplitude: np.float64 = zero + d.get('amplitude', 0.0)
        self.frequency: np.float64 = zero + d.get('frequency', 0.0)
        self.phase: np.float64 = zero + d.get('phase', 0.0)
        self.dc: np.float64 = zero + d.get('dc', dc_val)
        return

    def value(self, t) -> np.float64:
        if self.type == "sin":
            return self.amplitude*np.sin(2*np.pi*self.frequency*t+self.phase)+self.dc
        elif self.type == "dc":
            return self.dc
        else:
            return self.dc

    def __json__(self):
        return {
            "type": self.type,
            "amplitude": self.amplitude,
            "frequency": self.frequency,
            "phase": self.phase,
            "dc": self.dc,
        }

    def __repr__(self):
        return json.dumps(self.__json__(), indent=2)

    def __eq__(self, __o: 'NodeParamIback') -> bool:
        for key in NodeParamIback.Keys:
            if getattr(self, key) != getattr(__o, key):
                return False
        return True

    def __sub__(self, __o: 'NodeParamIback') -> dict:
        d = {}
        for key in NodeParamIback.Keys:
            if getattr(self, key) != getattr(__o, key):
                d[key] = getattr(self, key) - getattr(__o, key)
        return d


class NodeParamIext:
    Keys = ['type', 'height', 't_start', 't_end']

    def __init__(self, d: dict | float):
        dc_val = 0.0
        if d is None:
            d = {}
        if type(d) in [float, int]:
            dc_val = d
            d = {}
        # we use += to make sure number are all np.float64
        self.type = d.get('type', None)
        self.height: np.float64 = zero + d.get('height', dc_val)
        self.t_start: np.float64 = zero + d.get('t_start', 0.0)
        self.t_end: np.float64 = zero + d.get('t_end', 0.0)

    def value(self, t) -> np.float64:
        if self.type == "rectangular":
            if t >= self.t_start and t <= self.t_end:
                return self.height
            else:
                return zero
        else:
            return zero

    def __json__(self):
        return {
            "type": self.type,
            "height": self.height,
            "t_start": self.t_start,
            "t_end": self.t_end,
        }

    def __repr__(self):
        return json.dumps(self.__json__(), indent=2)

    def __eq__(self, __o: 'NodeParamIext') -> bool:
        for key in NodeParamIext.Keys:
            if getattr(self, key) != getattr(__o, key):
                return False
        return True

    def __sub__(self, __o: 'NodeParamIext') -> dict:
        d = {}
        for key in NodeParamIext.Keys:
            if getattr(self, key) != getattr(__o, key):
                d[key] = getattr(self, key) - getattr(__o, key)
        return d


class NodeParam:
    Keys = ['tau', 'sigma', 'gamma', 'tau_ampa',
            'sigma_ampa', 'gamma_ampa', 'I_back', 'I_ext']

    def __init__(self, d: dict):
        if d is None:
            d = {}
        self.tau = zero + d.get('tau', 0.0)
        self.sigma = zero + d.get('sigma', 0.0)
        self.gamma = zero + d.get('gamma', 0.0)
        self.tau_ampa = zero + d.get('tau_ampa', 0.0)
        self.sigma_ampa = zero + d.get('sigma_ampa', 0.0)
        self.gamma_ampa = zero + d.get('gamma_ampa', 0.0)
        self.I_back = NodeParamIback(d.get('I_back', None))
        self.I_ext = NodeParamIext(d.get('I_ext', None))
        return

    def __json__(self):
        return {
            "tau": self.tau,
            "sigma": self.sigma,
            "gamma": self.gamma,
            "tau_ampa": self.tau_ampa,
            "sigma_ampa": self.sigma_ampa,
            "gamma_ampa": self.gamma_ampa,
            "I_back": self.I_back.__json__(),
            "I_ext": self.I_ext.__json__(),
        }

    def __repr__(self):
        return json.dumps(self.__json__(), indent=2)

    def __eq__(self, __o: 'NodeParam') -> bool:
        for key in NodeParam.Keys:
            if getattr(self, key) != getattr(__o, key):
                return False
        return True

    def __sub__(self, __o: 'NodeParam') -> dict:
        d = {}
        for key in NodeParam.Keys:
            if getattr(self, key) != getattr(__o, key):
                d[key] = getattr(self, key) - getattr(__o, key)
        return d
