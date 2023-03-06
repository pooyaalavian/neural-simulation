import json
import numpy as np
from src.mixin import Mixin

zero = np.float64(0.0)


class ConnectionParam_source(Mixin):
    Keys = ['exc1', 'exc2', 'pv', 'sst1', 'sst2', 'vip1', 'vip2']

    def __init__(self, d: dict, *, delta=False) -> None:
        self.__delta = delta
        if d is None:
            d = {}

        self.exc1 = self.__npget__(d, 'exc1', self.__delta)
        self.exc2 = self.__npget__(d, 'exc2', self.__delta)
        self.pv = self.__npget__(d, 'pv', self.__delta)
        self.sst1 = self.__npget__(d, 'sst1', self.__delta)
        self.sst2 = self.__npget__(d, 'sst2', self.__delta)
        self.vip1 = self.__npget__(d, 'vip1', self.__delta)
        self.vip2 = self.__npget__(d, 'vip2', self.__delta)


class ConnectionParam(Mixin):
    Keys = ['exc1', 'exc2', 'pv', 'sst1', 'sst2', 'vip1', 'vip2']

    def __init__(self, d: dict, *, delta=False) -> None:
        self.__delta = delta
        if d is None:
            d = {}
        self.exc1 = ConnectionParam_source(d.get('exc1', None), delta=self.__delta)
        self.exc2 = ConnectionParam_source(d.get('exc2', None), delta=self.__delta)
        self.pv = ConnectionParam_source(d.get('pv', None), delta=self.__delta)
        self.sst1 = ConnectionParam_source(d.get('sst1', None), delta=self.__delta)
        self.sst2 = ConnectionParam_source(d.get('sst2', None), delta=self.__delta)
        self.vip1 = ConnectionParam_source(d.get('vip1', None), delta=self.__delta)
        self.vip2 = ConnectionParam_source(d.get('vip2', None), delta=self.__delta)

    def get(self, source_node: str, destination_node: str) -> np.float64:
        src = getattr(self, destination_node, None)
        if src is None:
            return zero
        return getattr(src, source_node, zero)
    
    def print_matrix(self, show_zeros=False) -> None:
        def gg(x):
            if show_zeros==False and x == 0:
                return f'{"":>10}'
            return f'{x:>10.3f}'
        
        print(f'{"":>10}', end='')
        for src in self.Keys:
            print(f'{src:>10}', end='')
        print()
        for dst in self.Keys:
            print(f'{dst:>10}', end='')
            for src in self.Keys:
                print(gg(self.get(src, dst)), end='')
            print()
        return
