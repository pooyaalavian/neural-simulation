from typing import Union
import numpy as np
from src import ParameterSet, ConstantsParam
from src.nodes import NodeParam

zero = np.float64(0.0)


def normalize(n):
    if -1e-50 < n < 1e-50:
        return n*0.0
    return n


class NodeState:
    def __init__(self, name, d: Union[dict, 'NodeState'], *, is_delta=False):
        self.is_delta = is_delta
        self._name = name
        if isinstance(d, NodeState):
            self.s = d.s
            self.s_ampa = d.s_ampa
            self.r = d.r
            self.y = d.y
        else:
            self.s: np.float64 = zero + d.get('s', 0.0)
            self.s_ampa: np.float64 = zero + d.get('s_ampa', 0.0)
            self.r: np.float64 = zero + d.get('r', 0.0)
            self.y: np.float64 = zero + d.get('y', 0.0)

        # variables starting with _ are not serialized/de-serialized
        # these are used to hold internal state
        # you can't initialize them here
        #  they are calculated in your custom State.calcDelta() method
        self._input = zero
        self._phi = zero

    def serialize(self):
        self.check_bounds()
        return np.array((self.s, self.s_ampa, self.r, self.y))

    def serialize_g(self, p: NodeParam):
        return np.array((0, 0, 0, p.sigma))

    def deserialize(self, arr: np.array):
        self.s = normalize(arr[0])
        self.s_ampa = normalize(arr[1])
        self.r = normalize(arr[2])
        self.y = normalize(arr[3])

        return np.delete(arr, [0, 1, 2, 3])

    def __add__(self, delta: 'NodeState'):
        if not delta.is_delta:
            raise ValueError("Can only add deltas to states")
        newState = NodeState(self)
        newState.s += delta.s
        newState.s_ampa += delta.s_ampa
        newState.r += delta.r
        newState.y += delta.y
        return newState

    def check_bounds(self):
        return
        for k in ['s', 's_ampa', 'r', 'y', '_input', '_phi']:
            v = getattr(self, k)
            if np.abs(v) > 1e99:
                raise ValueError(
                    f"State variable {self._name}.{k} is too large")


class State:
    Nodes = ['exc1', 'exc2', 'pv', 'sst1', 'sst2', 'vip1', 'vip2']

    def __init__(self, d: Union[dict, 'State'] = None, *, is_delta=False):
        self.is_delta = is_delta
        if isinstance(d, State):
            self.exc1 = NodeState('exc1', d.exc1, is_delta=is_delta)
            self.exc2 = NodeState('exc2', d.exc2, is_delta=is_delta)
            self.pv = NodeState('pv', d.pv, is_delta=is_delta)
            self.sst1 = NodeState('sst1', d.sst1, is_delta=is_delta)
            self.sst2 = NodeState('sst2', d.sst2, is_delta=is_delta)
            self.vip1 = NodeState('vip1', d.vip1, is_delta=is_delta)
            self.vip2 = NodeState('vip2', d.vip2, is_delta=is_delta)
            return
        if d is None:
            d = {}
        self.exc1 = NodeState('exc1', d.get('exc1', {}), is_delta=is_delta)
        self.exc2 = NodeState('exc2', d.get('exc2', {}), is_delta=is_delta)
        self.pv = NodeState('pv', d.get('pv', {}), is_delta=is_delta)
        self.sst1 = NodeState('sst1', d.get('sst1', {}), is_delta=is_delta)
        self.sst2 = NodeState('sst2', d.get('sst2', {}), is_delta=is_delta)
        self.vip1 = NodeState('vip1', d.get('vip1', {}), is_delta=is_delta)
        self.vip2 = NodeState('vip2', d.get('vip2', {}), is_delta=is_delta)
        return

    def initialize(self, q: ParameterSet):
        raise NotImplementedError("Not implemented")

    def calcDelta(self):
        raise NotImplementedError("Not implemented")

    def serialize(self):
        return np.concatenate([
            getattr(self, n).serialize() for n in State.Nodes
        ])
        # return np.concatenate((
        #     self.exc1.serialize(), self.exc2.serialize(),
        #     self.pv.serialize(),
        #     self.sst1.serialize(), self.sst2.serialize(),
        #     self.vip1.serialize(), self.vip2.serialize()
        # ))

    def serialize_g(self, p: ParameterSet):
        return np.concatenate([
            getattr(self, n).serialize_g(getattr(p, n)) for n in State.Nodes
        ])
        # return np.concatenate((
        #     self.exc1.serialize_g(p.exc1), self.exc2.serialize_g(p.exc2),
        #     self.pv.serialize_g(p.pv),
        #     self.sst1.serialize_g(p.sst1), self.sst2.serialize_g(p.sst2),
        #     self.vip1.serialize_g(p.vip1), self.vip2.serialize_g(p.vip2),
        # ))

    def deserialize(self, arr: np.array):
        for n in State.Nodes:
            arr = getattr(self, n).deserialize(arr)
        # arr = self.exc1.deserialize(arr)
        # arr = self.exc2.deserialize(arr)
        # arr = self.pv.deserialize(arr)
        # arr = self.sst1.deserialize(arr)
        # arr = self.sst2.deserialize(arr)
        # arr = self.vip1.deserialize(arr)
        # arr = self.vip2.deserialize(arr)
        if len(arr) > 0:
            raise ValueError("Array not empty")
        return self

    def __add__(self, delta: 'State'):
        if not delta.is_delta:
            raise ValueError("Can only add deltas to states")
        newState = State(self)
        newState.exc1 += delta.exc1
        newState.exc2 += delta.exc2
        newState.pv += delta.pv
        newState.sst1 += delta.sst1
        newState.sst2 += delta.sst2
        newState.vip1 += delta.vip1
        newState.vip2 += delta.vip2
        return newState
