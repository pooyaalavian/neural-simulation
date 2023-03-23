import numpy as np
import logging 
import time 
from datetime import datetime
import json
from pathlib import Path

from src import ParameterSet, Plot
from src import ModelBase as Model
from src.integral import itoint

# np.seterr(all='raise')
np.set_printoptions(precision=4, suppress=True, linewidth=200)
logging.basicConfig(level=logging.WARN)


def run(t_end, changes = {}, *, dt=0.001, path:Path=None):
    print(f'estimated time: {1.1 * t_end/dt / 1000} seconds')
    t_start = time.time()
    t = np.linspace(0, t_end, int(t_end / dt) + 1)
    y0 = Model()
    y0.initialize()
    params = ParameterSet("structure.json")
    params.batch_update(changes)
    params.J.print_matrix()
    params.J_ampa.print_matrix()
    print(json.dumps(params.__flat_json__(ignore_zeros=True), indent=2))
    if path is not None:
        params.save(path / 'params.json')
        params.saveDelta(path / 'params_delta.json',base_file='structure.json')
        params.saveDeltaHtml(path / 'params_delta.html',base_file='structure.json')

    def calc_g_static():
      sigma = y0.serialize_g(params)
      g_vector = sigma * params.constants.tau_y
      g_matrix = np.diag(g_vector)
      g_matrix = g_matrix[:,~np.all(g_matrix == 0, axis=0)]
      g_matrix = g_matrix * 1.0
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

experiments = [{"sst1.I_back.dc": 0.2,"sst2.I_back.dc": 0.2, "constants.ratio":0.5}]
if __name__ == '__main__':
    exp = 'test'
    for e in range(len(experiments)): 
        dt = datetime.now()
        folder = Path(f'img/{exp}/{dt.strftime("%Y-%m-%d")}/{dt.strftime("%H%M%S")}')
        folder.mkdir(parents=True)
        t, res = run(10, changes = experiments[e] , dt=0.001, path=folder)
        # exc1_r = np.array([x.exc1.r for x in res])
        # exc2_r = np.array([x.exc2.r for x in res])
        # pv_r = np.array([x.pv.r for x in res])
        plots = [
            Plot(['exc1.r'], t_start=0, t_end=10, title='Exc 1 Firing Rate', file=folder / 'r1.svg'),
            Plot(['exc2.r'], t_start=0, t_end=10, title='Exc 2 Firing Rate', file=folder / 'r2.svg'),
            Plot(['pv.r'],   t_start=0, t_end=10, title='PV Firing Rate',    file=folder / 'pv.svg'),
            Plot(['sst1.r'], t_start=0, t_end=10, title='SST 1 Firing Rate', file=folder / 's1.svg'),
            Plot(['sst2.r'], t_start=0, t_end=10, title='SST 2 Firing Rate', file=folder / 's2.svg'),
            Plot(['vip1.r'], t_start=0, t_end=10, title='VIP 1 Firing Rate', file=folder / 'v1.svg'),
            Plot(['vip2.r'], t_start=0, t_end=10, title='VIP 2 Firing Rate', file=folder / 'v2.svg'),
            # Plot(['exc1.r'], t_start=2, t_end=3, title='Exc 1 Firing Rate', file=f'{folder}/r1-before.png'),
            # Plot(['exc1.r'], t_start=5, t_end=6, title='Exc 1 Firing Rate', file=f'{folder}/r1-during.png'),
            # Plot(['exc1.r'], t_start=7, t_end=8, title='Exc 1 Firing Rate', file=f'{folder}/r1-after.png'),
            # Plot(['exc1.r','exc2.r','pv.r'], t_start=3, t_end=7, title='Exc 1, 2, PV Firing Rate', file=f'{folder}/r1-r2.png'),
        ]
        for p in plots:
            p(t,res)
        plots_ref = [p.file.name for p in plots]
        plots_ref = [f'<div class="res-img"><img src="{p}"/></div>' for p in plots_ref]
        plots_ref = '\n'.join(plots_ref)

        html = folder / 'results.html'
        html.write_text(f'''<html>
    <head>
        <title> Summary </title>
        <style>
        .images {{display: flex; flex-wrap: wrap;}}
        .res-img {{}}
        </style>
    </head>
    <body>
    <section id="inputs">
        <iframe src="params_delta.html" onload='javascript:(function(o){{o.style.height=o.contentWindow.document.body.scrollHeight+"px";}}(this));' 
        style="height:200px;width:100%;border:none;overflow:hidden;"></iframe>
    </section>
    <section id="results">
        <div class="images">{plots_ref}</div>
    </section>
    </body>
    </html>''')