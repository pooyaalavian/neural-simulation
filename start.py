import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import sdeint
# from pactools import Comodulogram, REFERENCES
# from pactools import simulate_pac

from src import ParameterSet
from src import ModelBase as Model
from src.integral import itoint
import json 
import logging 
import time 
# np.seterr(all='raise')
np.set_printoptions(precision=4, suppress=True, linewidth=300)
logging.basicConfig(level=logging.WARN)


def arr2str(arr: np.array):
    return ', '.join([f'{x:11.3f}' for x in arr])


def solve(f, g, y0: np.array, t: np.array):
    y = sdeint.itoint(f, g, y0, t)
    # return y
    def toState(y): return Model().deserialize(y)
    return list(map(toState, y))


def run(t_end, dt=0.001):
    print(f'estimated time: {1.1 * t_end/dt / 1000} seconds')
    t_start = time.time()
    t = np.linspace(0, t_end, int(t_end / dt) + 1)
    y0 = Model()
    y0.initialize()
    params = ParameterSet("structure.json")
    params.J.print_matrix()
    params.J_ampa.print_matrix()
    sigma = y0.serialize_g(params)
    g_vector = sigma * params.constants.tau_y
    g_matrix = np.diag(g_vector)
    g_matrix = g_matrix[:,~np.all(g_matrix == 0, axis=0)]
    g_matrix = -0.0 * g_matrix

    def model_f(y, t):
        Y = Model().deserialize(y)
        delta = Y.calcDelta(t, params)
        dy = delta.serialize()
        logging.debug(f' y: {arr2str(y)}')
        logging.debug(f'dy: {arr2str(dy)}')
        return dy

    def model_g(y, t):
        # Y = MyState().deserialize(y)
        # sigma = Y.serialize_g(params)
        # tau_y = params.constants.tau_y
        # g = sigma * tau_y
        # return np.diag(g_vector)
        return g_matrix.copy()

    gen = np.random.Generator(np.random.PCG64(123))
    # res = sdeint.itoint(model_f, model_g, y0.serialize(), t, gen)
    res = itoint(model_f, model_g, y0.serialize(), t, gen)
    def toState(y): return Model().deserialize(y)
    t_end = time.time()
    print(f'elapsed time: {t_end - t_start} seconds')
    return t, list(map(toState, res))




t, res = run(10, 0.001)
exc1_r = np.array([x.exc1.r for x in res])
exc2_r = np.array([x.exc2.r for x in res])
pv_r = np.array([x.pv.r for x in res])
plt.plot(t, exc1_r)
plt.plot(t, exc2_r)
plt.plot(t, pv_r)
with open('results.json', 'w') as f:
    data = {'t': t.tolist(), 
            'exc1_r': exc1_r.tolist(), 
            'exc2_r': exc2_r.tolist(),
            'pv_r': pv_r.tolist(),
          }
    json.dump(data, f)