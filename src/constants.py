import json
import numpy as np
from src.mixin import Mixin

zero = np.float64(0.0)


class ConstantsParam(Mixin):
    Keys = ['tau_r', 'tau_n','tau_y', 'a', 'b', 'd', 'g_I', 'c_1', 'c_0', 'r_0', 'ratio']

    def __init__(self, d: dict, *, delta=False) -> None:
        self.__delta = delta
        if d is None:
            d = {}
        self.tau_r = self.__npget__(d, 'tau_r', delta=self.__delta)
        self.tau_n = self.__npget__(d, 'tau_n', delta=self.__delta)
        self.tau_y = self.__npget__(d, 'tau_y', delta=self.__delta)
        self.a = self.__npget__(d, 'a', delta=self.__delta)
        self.b = self.__npget__(d, 'b', delta=self.__delta)
        self.d = self.__npget__(d, 'd', delta=self.__delta)
        self.g_I = self.__npget__(d, 'g_I', delta=self.__delta)
        self.c_1 = self.__npget__(d, 'c_1', delta=self.__delta)
        self.c_0 = self.__npget__(d, 'c_0', delta=self.__delta)
        self.r_0 = self.__npget__(d, 'r_0', delta=self.__delta)
        self.ratio = self.__npget__(d, 'ratio', delta=self.__delta)
        return
