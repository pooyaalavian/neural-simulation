import json
import numpy as np

zero = np.float64(0.0)


class ConnectionParam_source:
    Keys = ['exc1', 'exc2', 'pv', 'sst1', 'sst2', 'vip1', 'vip2']

    def __init__(self, d: dict) -> None:
        if d is None:
            d = {}

        self.exc1: np.float64 = zero + d.get('exc1', 0.0)
        self.exc2: np.float64 = zero + d.get('exc2', 0.0)
        self.pv: np.float64 = zero + d.get('pv', 0.0)
        self.sst1: np.float64 = zero + d.get('sst1', 0.0)
        self.sst2: np.float64 = zero + d.get('sst2', 0.0)
        self.vip1: np.float64 = zero + d.get('vip1', 0.0)
        self.vip2: np.float64 = zero + d.get('vip2', 0.0)

    def __json__(self):
        return {
            "exc1": self.exc1,
            "exc2": self.exc2,
            "pv": self.pv,
            "sst1": self.sst1,
            "sst2": self.sst2,
            "vip1": self.vip1,
            "vip2": self.vip2,
        }

    def __repr__(self):
        return json.dumps(self.__json__(), indent=2)

    def __eq__(self, other: 'ConnectionParam_source') -> bool:
        for key in ConnectionParam_source.Keys:
            if getattr(self, key) != getattr(other, key):
                return False
        return True

    def __sub__(self, other: 'ConnectionParam_source') -> dict:
        d = {}
        for key in ConnectionParam_source.Keys:
            if getattr(self, key) != getattr(other, key):
                d[key] = getattr(self, key) - getattr(other, key)
        return d


class ConnectionParam:
    Keys = ['exc1', 'exc2', 'pv', 'sst1', 'sst2', 'vip1', 'vip2']

    def __init__(self, d: dict):
        if d is None:
            d = {}
        self.exc1 = ConnectionParam_source(d.get('exc1', None))
        self.exc2 = ConnectionParam_source(d.get('exc2', None))
        self.pv = ConnectionParam_source(d.get('pv', None))
        self.sst1 = ConnectionParam_source(d.get('sst1', None))
        self.sst2 = ConnectionParam_source(d.get('sst2', None))
        self.vip1 = ConnectionParam_source(d.get('vip1', None))
        self.vip2 = ConnectionParam_source(d.get('vip2', None))

    def get(self, source_node: str, destination_node: str) -> np.float64:
        src = getattr(self, destination_node, None)
        if src is None:
            return zero
        return getattr(src, source_node, zero)

    def __json__(self):
        return {
            "exc1": self.exc1.__json__(),
            "exc2": self.exc2.__json__(),
            "pv": self.pv.__json__(),
            "sst1": self.sst1.__json__(),
            "sst2": self.sst2.__json__(),
            "vip1": self.vip1.__json__(),
            "vip2": self.vip2.__json__(),
        }

    def __repr__(self):
        return json.dumps(self.__json__(), indent=2)

    def __eq__(self, other: 'ConnectionParam') -> bool:
        for key in ConnectionParam.Keys:
            if getattr(self, key) != getattr(other, key):
                return False
        return True

    def __sub__(self, other: 'ConnectionParam') -> dict:
        d = {}
        for key in ConnectionParam.Keys:
            if getattr(self, key) != getattr(other, key):
                d[key] = getattr(self, key) - getattr(other, key)
        return d

