import numpy as np
import logging 
import time 

from src import ParameterSet, Plot
from src import ModelBase as Model
from src.integral import itoint

# np.seterr(all='raise')
np.set_printoptions(precision=4, suppress=True, linewidth=300)
logging.basicConfig(level=logging.WARN)


def run(t_end, dt=0.001):
    print(f'estimated time: {1.1 * t_end/dt / 1000} seconds')
    t_start = time.time()
    t = np.linspace(0, t_end, int(t_end / dt) + 1)
    y0 = Model()
    y0.initialize()
    params = ParameterSet("structure.json")
    params.J.print_matrix()
    params.J_ampa.print_matrix()

    def calc_g_static():
      sigma = y0.serialize_g(params)
      g_vector = sigma * params.constants.tau_y
      g_matrix = np.diag(g_vector)
      g_matrix = g_matrix[:,~np.all(g_matrix == 0, axis=0)]
      g_matrix = g_matrix * 10
      return g_matrix
    
    g_matrix = calc_g_static()

    def model_f(y, t):
        Y = Model().deserialize(y)
        delta = Y.calcDelta(t, params)
        dy = delta.serialize()
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


if __name__ == '__main__':
    t, res = run(10, 0.001)
    exc1_r = np.array([x.exc1.r for x in res])
    exc2_r = np.array([x.exc2.r for x in res])
    pv_r = np.array([x.pv.r for x in res])
    folder = 'img/exp1'
    plots = [
        Plot(['exc1.r'], t_start=2, t_end=3, title='Exc 1 Firing Rate', file=f'{folder}/r1-before.png'),
        Plot(['exc1.r'], t_start=5, t_end=6, title='Exc 1 Firing Rate', file=f'{folder}/r1-during.png'),
        Plot(['exc1.r'], t_start=7, t_end=8, title='Exc 1 Firing Rate', file=f'{folder}/r1-after.png'),
        Plot(['exc1.r','exc2.r','pv.r'], t_start=3, t_end=7, title='Exc 2 Firing Rate', file=f'{folder}/r1-r2.png'),
    ]
    for p in plots:
        p(t,res)
