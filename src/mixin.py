import json
import numpy as np


class Mixin:
    def __init__(self):
        self._mixin = True
        self.__delta = False

    def __json__(self):
        d = {}
        for key in self.Keys:
            v = getattr(self, key)
            if self.__ismixin__(v):
                d[key] = v.__json__()
            elif v is not None:
                d[key] = v
        return d

    def __repr__(self):
        return json.dumps(self.__json__(), indent=2)

    def __eq__(self, other) -> bool:
        for key in self.Keys:
            if getattr(self, key) != getattr(other, key):
                return False
        return True

    def __ismixin__(self, obj):
        try:
            return Mixin in obj.__class__.__bases__
        except:
            return False

    def __sub__(self, base):
        d = self.__class__({}, delta=True)
        # d = {}
        for key in self.Keys:
            current = getattr(self, key)
            other = getattr(base, key, None)
            if current != other:
                if self.__ismixin__(current):
                    value = current - other
                else:
                    value = current
                setattr(d, key, value)
            else:
                pass
        return d

    def __add__(self, second):
        d = self.__class__(self.__json__())
        if second.__delta == False:
            print('adding non-delta object')

        for key in self.Keys:
            current = getattr(self, key)
            other = getattr(second, key, None)
            # print(f'__add__ {key} {current} {other}')
            if current != other:
                if self.__ismixin__(current):
                    value = current + other
                elif other is not None:
                    value = other
                else:
                    value = current
                setattr(d, key, value)
        return d

    def __npget__(self, d, key, delta=False):
        v = d.get(key, None)
        if v is None:
            return None if delta else np.float64(0)
        return np.float64(v)

    def __flat_json__(self, *, ignore_zeros=False):
        d = {}
        for key in self.Keys:
            v = getattr(self, key)
            if self.__ismixin__(v):
                tmp = v.__flat_json__(ignore_zeros=ignore_zeros)
                for k in tmp:
                    d[f'{key}.{k}'] = tmp[k]
            else:
              if v is not None:
                  d[key] = v
        if ignore_zeros:
            for key in list(d.keys()):
                if d[key] == 0:
                    del d[key]
        return d

    def get(self, key):
        if key in self.Keys:
            return getattr(self, key)
        return None

    def __update__(self, address, value):
        if '.' in address:
            key, rest = address.split('.', 1)
            if key in self.Keys:
                v = getattr(self, key)
                if self.__ismixin__(v):
                    v.__update__(rest, value)
                else:
                    setattr(self, key, value)
        else:
            if address in self.Keys:
                setattr(self, address, value)
            else: 
                raise ValueError(f'Failed to set {address} to {value}')
        return

class A(Mixin):
    Keys = ['a', 'b', 'c', 'x', 'y']

    def __init__(self, d):
        super().__init__()
        for k in A.Keys:
            if k in d:
                setattr(self, k, d[k])
            else:
                setattr(self, k, None)
        return
