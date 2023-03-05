import numpy as np
from src.state import State, NodeState
from src.param import ParameterSet
from src.nodes import NodeParam
from src.constants import ConstantsParam


class ModelBase(State):

    def initialize(self):
        self.exc1.s = 0.5
        self.exc2.s = 0.5
        self.pv.s = 0.3
        self.exc1.s_ampa = 0.5
        self.exc2.s_ampa = 0.5
        self.sst1.s = 0.5
        self.sst2.s = 0.5
        self.vip1.s = 0.5
        self.vip2.s = 0.5

    def calc_inputs(self, t: np.float64, p: ParameterSet):
        for dest in self.Nodes:
            n_dest: NodeState = getattr(self, dest)
            p_dest: NodeParam = getattr(p, dest)
            # add I_background and I_external to input
            n_dest._input += p_dest.I_back.value(t)
            n_dest._input += p_dest.I_ext.value(t)
            # add current state's "y" to input
            n_dest._input += n_dest.y

            # for each node in graph, clacl flow to this node
            for src in self.Nodes:
                n_src: NodeState = getattr(self, src)
                p_src: NodeParam = getattr(p, src)
                # TODO: do we need to skip src==dest?
                # if src == dest:
                #     continue
                n_dest._input += n_src.s * \
                    p.J.get(src, dest) + n_src.s_ampa * p.J_ampa.get(src, dest)

        # adjust for p.constants.ratio
        self.sst1._input *= (1-p.constants.ratio)
        self.sst2._input *= (1-p.constants.ratio)
        return

    @staticmethod
    def phi(n: NodeState, p: ConstantsParam) -> np.float64:
        v = (p.c_1 * n._input - p.c_0) / p.g_I + p.r_0
        # to keep it as a numpy float
        if v < 0:
            v *= 0
        return v

    def calc_phis(self, t: np.float64, p: ParameterSet):
        for dest in self.Nodes:
            n_dest: NodeState = getattr(self, dest)
            n_dest._phi = self.phi(n_dest, p.constants)

    def calc_ds(self, delta: 'ModelBase', t: np.float64, params: ParameterSet):
        for node_name in self.Nodes:
            n: NodeState = getattr(self, node_name)
            p: NodeParam = getattr(params, node_name)
            d: NodeState = getattr(delta, node_name)

            s_tau = - (n.s / p.tau)
            gamma_r = p.gamma * n.r
            if node_name in ['exc1', 'exc2']:
                gamma_r *= (1-n.s)
            d.s = s_tau + gamma_r

            if node_name in ['exc1', 'exc2']:
                s_tau_ampa = - (n.s_ampa / p.tau_ampa)
                gamma_r_ampa = p.gamma_ampa * n.r
                d.s_ampa = s_tau_ampa + gamma_r_ampa

        return

    def calc_dr(self, delta: 'ModelBase', t: np.float64, params: ParameterSet):
        for node_name in self.Nodes:
            n: NodeState = getattr(self, node_name)
            p: NodeParam = getattr(params, node_name)
            d: NodeState = getattr(delta, node_name)
            # a new variable gamma_r is referenced here
            # TODO: add to parameterSet.NodeParams?
            # d.r = (n._phi - n.r)/params.constants.tau_r + p.gamma_r
            Warning("gamma_r is not a parameter... using gamma insteads")
            d.r = (n._phi - n.r)/params.constants.tau_r + p.gamma
        return

    def calc_dy(self, delta: 'ModelBase', t: np.float64, params: ParameterSet):
        for node_name in self.Nodes:
            n: NodeState = getattr(self, node_name)
            d: NodeState = getattr(delta, node_name)
            d.y = n.y / params.constants.tau_n
        return

    def calcDelta(self, t: np.float64, params: ParameterSet):
        self.calc_inputs(t, params)
        self.calc_phis(t, params)

        delta = self.__class__(is_delta=True)
        self.calc_ds(delta, t, params)
        self.calc_dr(delta, t, params)
        self.calc_dy(delta, t, params)

        return delta
