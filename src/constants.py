import json
import numpy as np

zero = np.float64(0.0)


class ConstantsParam:
    Keys = ['tau_r', 'tau_n', 'a', 'b', 'd', 'g_I', 'c_1', 'c_0', 'r_0']

    def __init__(self, d: dict):
        if d is None:
            d = {}
        self.tau_r: np.float64 = zero + d.get('tau_r', 0.0)
        self.tau_n: np.float64 = zero + d.get('tau_n', 0.0)
        self.a: np.float64 = zero + d.get('a', 0.0)
        self.b: np.float64 = zero + d.get('b', 0.0)
        self.d: np.float64 = zero + d.get('d', 0.0)
        self.g_I: np.float64 = zero + d.get('g_I', 0.0)
        self.c_1: np.float64 = zero + d.get('c_1', 0.0)
        self.c_0: np.float64 = zero + d.get('c_0', 0.0)
        self.r_0: np.float64 = zero + d.get('r_0', 0.0)
        return

    def __json__(self):
        return {
            "tau_r": self.tau_r,
            "tau_n": self.tau_n,
            "a": self.a,
            "b": self.b,
            "d": self.d,
            "g_I": self.g_I,
            "c_1": self.c_1,
            "c_0": self.c_0,
            "r_0": self.r_0,
        }

    def __repr__(self) -> str:
        return json.dumps(self.__json__(), indent=2)

    def __eq__(self, __o: 'ConstantsParam') -> bool:
        for key in ConstantsParam.Keys:
            if getattr(self, key) != getattr(__o, key):
                return False
        return True

    def __sub__(self, __o: 'ConstantsParam') -> dict:
        d = {}
        for key in ConstantsParam.Keys:
            if getattr(self, key) != getattr(__o, key):
                d[key] = getattr(self, key) - getattr(__o, key)
        return d
