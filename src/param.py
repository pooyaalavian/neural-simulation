import json
from src.mixin import Mixin
from src.nodes import NodeParam
from src.connections import ConnectionParam
from src.constants import ConstantsParam
import warnings


class ParameterSet(Mixin):
    Keys = ['exc1', 'exc2', 'pv', 'sst1', 'sst2',
            'vip1', 'vip2', 'J', 'J_ampa', 'constants']

    def __init__(self, filename_or_dict, *, delta=False):
        self.__delta = delta
        if type(filename_or_dict) == str:
            self.base_file = filename_or_dict
            self.base_dict = {}
            f = open(self.base_file, 'r')
            d = json.load(f)
        else:
            self.base_file = None
            self.base_dict = filename_or_dict
            d = self.base_dict

        self.exc1 = NodeParam(d.get('exc1', None), delta=self.__delta)
        self.exc2 = NodeParam(d.get('exc2', None), delta=self.__delta)
        self.pv   = NodeParam(d.get('pv',   None), delta=self.__delta)
        self.sst1 = NodeParam(d.get('sst1', None), delta=self.__delta)
        self.sst2 = NodeParam(d.get('sst2', None), delta=self.__delta)
        self.vip1 = NodeParam(d.get('vip1', None), delta=self.__delta)
        self.vip2 = NodeParam(d.get('vip2', None), delta=self.__delta)
        self.J = ConnectionParam(d.get('J', None), delta=self.__delta)
        self.J_ampa = ConnectionParam(d.get('J_ampa', None), delta=self.__delta)
        self.constants = ConstantsParam(d.get('constants', None), delta=self.__delta)
        self.recalculate()
        return

    def recalculate(self):
        # todo: calc J_pve1, J_pve2
        c = self.constants
        J = self.J
        p = self.pv
        eta = p.tau * p.gamma * c.c_1/(
            c.g_I - J.pv.pv * p.tau * p.gamma * c.c_1)
        J_s = J.exc1.exc1
        J_c = J.exc1.exc2
        J_ei= J.exc1.pv
        if J.exc1.exc1 != J.exc2.exc2:
            warnings.warn('Calculating J matrix: J.exc1.exc1 != J.exc2.exc2 -- J_s calculated as average')
            J_s = (J.exc1.exc1 + J.exc2.exc2)/2.0 
        if J.exc1.exc2 != J.exc2.exc1:
            warnings.warn('Calculating J matrix: J.exc1.exc2 != J.exc2.exc1 -- J_c calculated as average')
            J_c = (J.exc1.exc2 + J.exc2.exc1)/2.0
        if J.exc1.pv != J.exc2.pv:
            warnings.warn('Calculating J matrix: J.exc1.pv != J.exc2.pv -- J_ei calculated as average')
            J_ei = (J.exc1.pv + J.exc2.pv)/2.0
        J_ie = (c.J_0 - J_s - J_c)/(2 * J_ei * eta)
        self.J.pv.exc1 = J_ie
        self.J.pv.exc2 = J_ie
        return

    def getDelta(self, *, base_file: str = None):
        if base_file is None:
            base_file = self.base_file
        base = ParameterSet(base_file)
        d = self - base
        return d

    def save(self, filename: str):
        with open(filename, 'w') as f:
            json.dump(str(self), f, indent=2)

    def saveDelta(self, filename: str, *, base_file: str):
        delta = self.getDelta(base_file)
        with open(filename, 'w') as f:
            json.dump(delta, f, indent=2)
