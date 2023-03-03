import json
from src.nodes import NodeParam
from src.connections import ConnectionParam
from src.constants import ConstantsParam


class ParameterSet:
    Keys = ['exc1', 'exc2', 'pv', 'sst1', 'sst2',
            'vip1', 'vip2', 'J', 'J_ampa', 'constants']

    def __init__(self, filename):
        loaded = False
        with open(filename, 'r') as f:
            d = json.load(f)
            self.exc1 = NodeParam(d.get('exc1', None))
            self.exc2 = NodeParam(d.get('exc2', None))
            self.pv = NodeParam(d.get('pv', None))
            self.sst1 = NodeParam(d.get('sst1', None))
            self.sst2 = NodeParam(d.get('sst2', None))
            self.vip1 = NodeParam(d.get('vip1', None))
            self.vip2 = NodeParam(d.get('vip2', None))
            self.J = ConnectionParam(d.get('J', None))
            self.J_ampa = ConnectionParam(d.get('J_ampa', None))
            self.constants = ConstantsParam(d.get('constants', None))
            loaded = True
        if not loaded:
            raise ValueError("Could not load parameter set from file")

        # todo: calc J_pve1, J_pve2
        # self.J.pv.exc1 = 0.0
        # self.J.pv.exc2 = 0.0

    def __json__(self):
        return {
            "exc1": self.exc1.__json__(),
            "exc2": self.exc2.__json__(),
            "pv": self.pv.__json__(),
            "sst1": self.sst1.__json__(),
            "sst2": self.sst2.__json__(),
            "vip1": self.vip1.__json__(),
            "vip2": self.vip2.__json__(),
            "J": self.J.__json__(),
            "J_ampa": self.J_ampa.__json__(),
            "constants": self.constants.__json__(),
        }

    def __repr__(self):
        return json.dumps(self.__json__(), indent=2)

    def __eq__(self, __o: 'ParameterSet') -> bool:
        for key in ParameterSet.Keys:
            if getattr(self, key) != getattr(__o, key):
                return False
        return True

    def __sub__(self, __o: 'ParameterSet') -> dict:
        d = {}
        o=__o
        for key in ParameterSet.Keys:
            if getattr(self, key) != getattr(o, key):
                d[key] = getattr(self, key) - getattr(o, key)
        return d

    def getDelta(self, base_file: str='defaults.json'):
        base = ParameterSet(base_file)
        d = self - base
        return d
    
    def save(self, filename: str):
        with open(filename, 'w') as f:
            json.dump(str(self), f, indent=2)

    def saveDelta(self, filename: str, base_file: str):
        delta = self.getDelta(base_file)
        with open(filename, 'w') as f:
            json.dump(delta, f, indent=2)
