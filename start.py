import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import sdeint
# from pactools import Comodulogram, REFERENCES
# from pactools import simulate_pac

from src import ParameterSet
from src import ModelBase as Model
import logging 
np.seterr(all='raise')
logging.basicConfig(level=logging.DEBUG)




def solve(f, g, y0: np.array, t: np.array):
    y = sdeint.itoint(f, g, y0, t)
    # return y
    def toState(y): return Model().deserialize(y)
    return list(map(toState, y))


def run(t_end, dt=0.0001):
    print(f'estimated time: {1.1 * t_end/dt / 1000} seconds')
    t = np.linspace(0, t_end, int(t_end / dt) + 1)
    y0 = Model()
    y0.initialize()
    params = ParameterSet("structure.json")
    params.J.print_matrix()
    params.J_ampa.print_matrix()
    sigma = y0.serialize_g(params)
    g_vector = sigma * params.constants.tau_y

    def model_f(y, t):
        Y = Model().deserialize(y)
        delta = Y.calcDelta(t, params)
        return delta.serialize()

    def model_g(y, t):
        # Y = MyState().deserialize(y)
        # sigma = Y.serialize_g(params)
        # tau_y = params.constants.tau_y
        # g = sigma * tau_y
        return np.diag(g_vector)

    res = sdeint.itoint(model_f, model_g, y0.serialize(), t)
    def toState(y): return Model().deserialize(y)
    return t, list(map(toState, res))


t, res = run(20)
exc1_r = np.array([x.exc1.r for x in res])
exc2_r = np.array([x.exc2.r for x in res])
pv_r = np.array([x.pv.r for x in res])
plt.plot(t, exc1_r)
plt.plot(t, exc2_r)
plt.plot(t, pv_r)
