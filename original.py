import numpy as np
from scipy.integrate import odeint
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import sys
from pytictoc import TicToc
import math
from random import gauss
from random import seed
from pandas import Series
import sdeint
from numpy import linalg as LA
import itertools
import json
# from pactools import Comodulogram, REFERENCES
# from pactools import simulate_pac
# np.seterr(all='raise')
np.set_printoptions(precision=4, suppress=True, linewidth=300)



# file = open('original.out', 'w')
# print_org = print
# def print(*args, **kwargs):
#     print_org(*args, file=file, **kwargs)


def arr2str(arr):
    return ', '.join([f'{x:12.3f}' for x in arr])


# I_i = np.linspace(0.4,0.6,11)
# I_e = np.linspace(0.2,0.6,11)
# I_i=np.array([0.2,0.3,0.35,0.4,0.5])
# I_e =np.array([0.4])
# inputs = list(itertools.product(I_i,I_e))
# version 2
# I_i=np.array([0.4])
# I_e =np.array([0.3,0.39,0.5])
# inputs = list(itertools.product(I_e,I_i))
# version 3
I_i = np.array([0.35, 45])
I_e = np.array([0.4])
inputs = list(itertools.product(I_e, I_i))
# singles
I_e = [0.45]
I_i = [0.35]
inputs = list(itertools.product(I_e, I_i))


def close(func, *args):
    def newfunc(x, t):
        return func(x, t, *args)
    return newfunc


def arr2str(arr):
    return ', '.join([f'{x:12.3f}' for x in arr])


def g_fun(z, t, P):
    sigma_a = P[0]
    sigma_b = P[1]
    sigma_c = P[2]
    sigma_d = P[3]
    sigma_e = P[4]
    sigma_f = P[5]
    sigma_g = P[6]

    tau_y = P[7]
    # tau_y=0

    x = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, sigma_a*tau_y, sigma_b *
         tau_y, sigma_c*tau_y, sigma_d*tau_y, sigma_e*tau_y, sigma_f*tau_y, sigma_g*tau_y]
    # print(len(x),x)
    g_matrix = np.diag(x)
    g_matrix = g_matrix[:,~np.all(g_matrix == 0, axis=0)]
    return g_matrix


printed = False


def model_MW(z, t, Q):
    def printInfo():
        global printed
        if printed:
            return
        printed = True
        print("tau_a = ", tau_a)
        print("tau_b = ", tau_b)
        print("tau_c = ", tau_c)
        print("tau_Aa = ", tau_Aa)
        print("tau_Ab = ", tau_Ab)
        print("tau_r = ", tau_r)
        print("tau_n = ", tau_n)
        print("tau_sa = ", tau_sa)
        print("tau_sb = ", tau_sb)
        print("tau_va = ", tau_va)
        print("tau_vb = ", tau_vb)

        print("gamma_a = ", gamma_a)
        print("gamma_b = ", gamma_b)
        print("gamma_Aa = ", gamma_Aa)
        print("gamma_Ab = ", gamma_Ab)
        print("gamma_c = ", gamma_c)
        print("gamma_sa = ", gamma_sa)
        print("gamma_sb = ", gamma_sb)
        print("gamma_va = ", gamma_va)
        print("gamma_vb = ", gamma_vb)

        print("gamma_ra = ", gamma_ra)
        print("gamma_rb = ", gamma_rb)
        print("gamma_rc = ", gamma_rc)
        print("gamma_rs = ", gamma_rs)
        print("gamma_v = ", gamma_v)

        print("p1 = ", p1)
        print("p2 = ", p2)

        print("g_I = ", g_I)
        print("c_1 = ", c_1)
        print("c_0 = ", c_0)
        print("r_0 = ", r_0)

        print("a = ", a)
        print("b = ", b)
        print("d = ", d)

        print("J_c = ", J_c)
        print("J_ei = ", J_ei)
        print("J_ii = ", J_ii)
        print("J_0 = ", J_0)
        print("J_s = ", J_s)
        print("J_se = ", J_se)
        print("J_ce = ", J_ce)
        print("J_sp = ", J_sp)
        print("J_es = ", J_es)
        print("J_pv = ", J_pv)
        print("J_sv = ", J_sv)
        print("J_vs = ", J_vs)
        print("J_Aa = ", J_Aa)
        print("J_Ab = ", J_Ab)
        return
    ratio = 0

    tau_a = Q[0]
    tau_b = Q[1]
    tau_c = Q[2]
    tau_Aa = Q[3]
    tau_Ab = Q[4]
    tau_r = Q[5]
    tau_n = Q[6]
    tau_sa = Q[7]
    tau_sb = Q[8]
    tau_va = Q[9]
    tau_vb = Q[10]

    gamma_a = Q[11]
    gamma_b = Q[12]
    gamma_Aa = Q[13]
    gamma_Ab = Q[14]
    gamma_c = Q[15]
    gamma_sa = Q[16]
    gamma_sb = Q[17]
    gamma_va = Q[18]
    gamma_vb = Q[19]

    gamma_ra = Q[20]
    gamma_rb = Q[21]
    gamma_rc = Q[22]
    gamma_rs = Q[23]
    gamma_v = Q[24]

    g_I = Q[25]
    c_1 = Q[26]
    c_0 = Q[27]
    r_0 = Q[28]

    a = Q[29]
    b = Q[30]
    d = Q[31]

    J_c = Q[32]
    J_ei = Q[33]
    J_ii = Q[34]

    J_0 = Q[35]

    J_s = Q[36]

    J_se = Q[37]
    J_ce = Q[38]
    J_sp = Q[39]

    J_es = Q[40]

    J_pv = Q[41]
    J_sv = Q[42]

    J_vs = Q[43]

    J_Aa = Q[44]
    J_Ab = Q[45]

    J_Ia = Q[46]
    J_Ib = Q[47]

    p1 = Q[48]
    p2 = Q[49]

    Iback_a = Q[50]
    Iback_b = Q[51]
    Iback_c = Q[52]

    Iback_d = Q[53]
    Iback_e = Q[54]

    Iback_f = Q[55]
    Iback_g = Q[56]

    fq = 8*math.pi  # 4HZ

    ca_F = 0.1
    ca_G = 0.1

    Iback_a += ca_F*np.sin(fq*t)
    Iback_b += ca_G*np.sin(fq*t)

    # Iback_f += ca_F*np.sin(fq*t)
    # Iback_g += ca_G*np.sin(fq*t)

    I_a = Q[57]
    I_b = Q[58]
    I_c = Q[59]

    I_va = Q[60]
    I_vb = Q[61]

    eta = tau_c*gamma_c*c_1/(g_I-J_ii*tau_c*gamma_c*c_1)
    J_ie = (J_0-J_s-J_c)/(2*J_ei*eta)

    idx = (t > 4 and t < 6)
    idy = (t > 12 and t < 14)

    # uncomment for input
    # I_va = I_va*idy
    # I_vb = I_va*idy

    I_a = I_a*idx
    I_b = I_b*idx
    I_c = I_c*idx

    # -------------------------------------------------------------
    print(f't: {t:.8f}')

    dz = np.zeros(23)

    sa = z[0]
    sb = z[1]
    sc = z[2]
    s_Aa = z[3]
    s_Ab = z[4]
    st_a = z[5]
    st_b = z[6]
    s_va = z[7]
    s_vb = z[8]
    r_a = z[9]
    r_b = z[10]
    r_c = z[11]
    r_d = z[12]
    r_e = z[13]
    r_f = z[14]
    r_g = z[15]
    y_a = z[16]
    y_b = z[17]
    y_c = z[18]
    y_d = z[19]
    y_e = z[20]
    y_f = z[21]
    y_g = z[22]
    _s = [sa, sb, sc, st_a, st_b, s_va, s_vb]
    _s_ampa = [s_Aa, s_Ab, 0, 0, 0, 0, 0]
    _r = [r_a, r_b, r_c, r_d, r_e, r_f, r_g]
    _y = [y_a, y_b, y_c, y_d, y_e, y_f, y_g]

    input_a  = y_a + Iback_a + I_a + J_s * sa + J_c * sb + J_ei* sc + J_Aa* s_Aa + J_Ab*s_Ab + J_es* st_a + 0.0 * st_b
    input_b  = y_b + Iback_b + I_b + J_c * sa + J_s * sb + J_ei* sc + J_Ab* s_Aa + J_Aa*s_Ab + 0.0 * st_a + J_es* st_b
    input_c  = y_c + Iback_c + I_c + J_ie* sa + J_ie* sb + J_ii* sc + J_Ia* s_Aa + J_Ib*s_Ab + 0.0 * st_a + 0.0 * st_b
    input_sa = y_d + Iback_d + 0.0 + J_se* sa + J_ce* sb + 0.0 * sc + J_sp* s_Aa + 0.0 *s_Ab + 0.0 * st_a + 0.0 * st_b + J_vs*s_va 
    input_sb = y_e + Iback_e + 0.0 + J_ce* sa + J_se* sb + 0.0 * sc + 0.0 * s_Aa + J_sp*s_Ab + 0.0 * st_a + 0.0 * st_b + J_vs*s_vb 
    input_va = y_f + Iback_f + I_va+ J_pv* sa + 0.0 * sb + 0.0 * sc + 0.0 * s_Aa + 0.0 *s_Ab + J_sv* st_a + 0.0 * st_b
    input_vb = y_g + Iback_g + I_vb+ 0.00* sa + J_pv* sb + 0.0 * sc + 0.0 * s_Aa + 0.0 *s_Ab + 0.0 * st_a + J_sv* st_b  

    # print(f'''J=
    # exc1: {input_a :12.3f} = {J_s :10.3f} sa + {J_c :12.3f} sb + {J_ei:12.3f} sc + {J_Aa:12.3f} sAa + {J_Ab:12.3f} sAb +                                       + {J_es:12.3f} s_st1                      + {Iback_a} Iback + {I_a}  I + {y_a} y
    # exc2: {input_b :12.3f} = {J_c :10.3f} sa + {J_s :12.3f} sb + {J_ei:12.3f} sc + {J_Ab:12.3f} sAa + {J_Aa:12.3f} sAb +                                                            + {J_es:12.3f} s_st2 + {Iback_b} Iback + {I_b}  I + {y_b} y
    # pv  : {input_c :12.3f} = {J_ie:10.3f} sa + {J_ie:12.3f} sb + {J_ii:12.3f} sc + {J_Ia:12.3f} sAa + {J_Ib:12.3f} sAb +                                                                                 + {Iback_c} Iback + {I_c}  I + {y_c} y
    # sst1: {input_sa:12.3f} = {J_se:10.3f} sa + {J_ce:12.3f} sb                   + {J_sp:12.3f} sAa                    + {J_vs:12.3f} s_va                                                               + {Iback_d} Iback            + {y_d} y
    # sst2: {input_sb:12.3f} = {J_ce:10.3f} sa + {J_se:12.3f} sb                                      + {J_sp:12.3f} sAb                     + {J_vs:12.3f} s_vb                                           + {Iback_e} Iback            + {y_e} y
    # vip1: {input_va:12.3f} = {J_pv:10.3f} sa                                                                                                                   + {J_sv:12.3f} s_st1                      + {Iback_f} Iback + {I_va} I + {y_f} y
    # vip2: {input_vb:12.3f} =                   {J_pv:12.3f} sb                                                                                                                      + {J_sv:12.3f} s_st2 + {Iback_g} Iback + {I_vb} I + {y_g} y
    # ''')
    # print(f'''
    # input_a:  {input_a :>8.3f}  sa: {sa:12.3f}
    # input_b:  {input_b :>8.3f}  sb: {sb:12.3f}
    # input_c:  {input_c :>8.3f}  sc: {sc:12.3f}
    #                     s_Aa: {s_Aa:12.3f}
    #                     s_Ab: {s_Ab:12.3f}
    # input_va: {input_va:>8.3f}   s_va: {s_va:12.3f}
    # input_vb: {input_vb:>8.3f}   s_vb: {s_vb:12.3f}
    # input_sa: {input_sa:>8.3f}   st_a: {st_a:12.3f}
    # input_sb: {input_sb:>8.3f}   st_b: {st_b:12.3f}
    # ''')

    inp_opto_sa = ratio*input_sa
    inp_opto_sb = ratio*input_sb
    input_sa -= inp_opto_sa
    input_sb -= inp_opto_sb

    _inputs = [
        input_a, input_b,
        input_c,
        input_sa, input_sb,
        input_va, input_vb,
    ]

    g1 = 1

    phi_a = g1*p1*(a*input_a-b)/(1-np.exp(-d*(a*input_a-b)))
    phi_b = g1*p2*(a*input_b-b)/(1-np.exp(-d*(a*input_b-b)))

    g1 = int(input_c >= (c_0-r_0*g_I)/c_1)
    phi_c = g1*(((c_1*input_c-c_0)/g_I)+r_0)

    g1 = int(input_sa >= (c_0-r_0*g_I)/c_1)
    phi_sa = g1*(((c_1*input_sa-c_0)/g_I)+r_0)

    g1 = int(input_sb >= (c_0-r_0*g_I)/c_1)
    phi_sb = g1*(((c_1*input_sb-c_0)/g_I)+r_0)

    g1 = int(input_va >= (c_0-r_0*g_I)/c_1)
    phi_va = g1*(((c_1*input_va-c_0)/g_I)+r_0)

    g1 = int(input_vb >= (c_0-r_0*g_I)/c_1)
    phi_vb = g1*(((c_1*input_vb-c_0)/g_I)+r_0)

    _phi = [
        phi_a, phi_b,
        phi_c,
        phi_sa, phi_sb,
        phi_va, phi_vb,
    ]

    dsa   = (-(sa  /tau_a ) +  gamma_a*r_a*(1-sa) )
    dsb   = (-(sb  /tau_b ) +  gamma_b*r_b*(1-sb) )
    dsc   = (-(sc  /tau_c ) +  gamma_c*r_c)
    ds_Aa = (-(s_Aa/tau_Aa) + gamma_Aa*r_a)
    ds_Ab = (-(s_Ab/tau_Ab) + gamma_Ab*r_b)
    dst_a = (-(st_a/tau_sa) + gamma_sa*r_d)
    dst_b = (-(st_b/tau_sb) + gamma_sb*r_e)
    ds_va = (-(s_va/tau_va) + gamma_va*r_f)
    ds_vb = (-(s_vb/tau_vb) + gamma_vb*r_g)

    _ds = [
        dsa, dsb,
        dsc,
        dst_a, dst_b,
        ds_va, ds_vb,
    ]
    _dsA = [
        ds_Aa, ds_Ab,
        0, 0, 0, 0, 0,
    ]
    dr_a = ((phi_a-r_a)/tau_r)*gamma_ra
    dr_b = ((phi_b-r_b)/tau_r)*gamma_rb
    dr_c = ((phi_c-r_c)/tau_r)*gamma_rc
    dr_d = ((phi_sa-r_d)/tau_r)*gamma_rs
    dr_e = ((phi_sb-r_e)/tau_r)*gamma_rs
    dr_f = ((phi_va-r_f)/tau_r)*gamma_v
    dr_g = ((phi_vb-r_g)/tau_r)*gamma_v
    _dr = [
        dr_a, dr_b,
        dr_c,
        dr_d, dr_e,
        dr_f, dr_g,
    ]
    # print('gamma_r: ',gamma_ra,gamma_rb,gamma_rc,gamma_rs,gamma_rs,gamma_v,gamma_v)
    # print('tau_n: ',tau_n)
    dy_a = -(y_a/tau_n)
    dy_b = -(y_b/tau_n)
    dy_c = -(y_c/tau_n)
    dy_d = -(y_d/tau_n)
    dy_e = -(y_e/tau_n)
    dy_f = -(y_f/tau_n)
    dy_g = -(y_g/tau_n)

    _dy = [
        dy_a, dy_b,
        dy_c,
        dy_d, dy_e,
        dy_f, dy_g,
    ]

    dz[0] = dsa
    dz[1] = dsb
    dz[2] = dsc
    dz[3] = ds_Aa
    dz[4] = ds_Ab
    dz[5] = dst_a
    dz[6] = dst_b
    dz[7] = ds_va
    dz[8] = ds_vb
    dz[9] = dr_a
    dz[10] = dr_b
    dz[11] = dr_c
    dz[12] = dr_d
    dz[13] = dr_e
    dz[14] = dr_f
    dz[15] = dr_g
    dz[16] = dy_a
    dz[17] = dy_b
    dz[18] = dy_c
    dz[19] = dy_d
    dz[20] = dy_e
    dz[21] = dy_f
    dz[22] = dy_g
    printInfo()
    # print('-'*60)
    # print(f't: {t}')
    # print(f's:     {arr2str(_s)}')
    # print(f'sA:    {arr2str(_s_ampa)}')
    # print(f'r:     {arr2str(_r)}')
    # print(f'y:     {arr2str(_y)}')
    # print(f'input: {arr2str(_inputs)}')
    # print(f'phi:   {arr2str(_phi)}')
    # print(f'ds:    {arr2str(_ds)}')
    # print(f'dsA:   {arr2str(_dsA)}')
    # print(f'dr:    {arr2str(_dr)}')
    # print(f'dy:    {arr2str(_dy)}')
    # print('*'*60)

    print('[z]:  ', arr2str(z))
    print('[dz]: ', arr2str(dz))
    res = {
        't': f'{t:.8f}',
        'z': {
            's': [f'{x:.8f}' for x in _s],
            's_ampa': [f'{x:.8f}' for x in _s_ampa],
            'r': [f'{x:.8f}' for x in _r],
            'y': [f'{x:.8f}' for x in _y],
            'input': [f'{x:.8f}' for x in _inputs],
            'phi': [f'{x:.8f}' for x in _phi],
        },
        'dz': {
            'ds': [f'{x:.8f}' for x in _ds],
            'ds_ampa': [f'{x:.8f}' for x in _dsA],
            'dr': [f'{x:.8f}' for x in _dr],
            'dy': [f'{x:.8f}' for x in _dy],
        },
    }
    # print('json: ',json.dumps(res))

    return dz


def main():
    inp = 0

    # Parameter for noise
    sigma_a = 6  # 2.5
    sigma_b = 6
    sigma_c = 2.5
    sigma_d = 2.5
    sigma_e = 2.5
    sigma_f = 2.5
    sigma_g = 2.5
    tau_y = 0.01

    P = [[sigma_a, sigma_b, sigma_c, sigma_d, sigma_e, sigma_f, sigma_g, tau_y]]

    # Setting the background current

    # pp1 = 0.48
    # pp2 = 0.40

    # pp1 = 0.48
    # pp2 = 0.70

    # pp1 = 0.48
    # pp2 = 0.50

    # pp1 = 0.40
    # pp2 = 0.50

    # pp1 = 0.60
    # pp2 = 0.50

    pp1 = inputs[inp][0]
    pp2 = inputs[inp][1]
    print(pp1, pp2)

    # Parameters for the system
    tau_a = 0.0288
    tau_b = 0.0288
    tau_c = 0.0024
    tau_Aa = 0.00096
    tau_Ab = 0.00096
    tau_r = 0.002
    tau_n = 0.00096
    tau_sa = 0.0072
    tau_sb = 0.0072
    tau_va = 0.0072
    tau_vb = 0.0072

    gamma_a = 2.675
    gamma_b = 2.675
    gamma_Aa = 2.075
    gamma_Ab = 2.075
    gamma_c = 4.1625
    gamma_sa = 2.075
    gamma_sb = 2.075
    gamma_va = 2.075
    gamma_vb = 2.075
    # good

    gamma_ra = 1.0
    gamma_rb = 1.0
    gamma_rc = 1.0
    gamma_rs = 1.0
    gamma_v = 1.0

    g_I = 4
    c_1 = 615
    c_0 = 177
    r_0 = 5.5

    a = 135
    b = 54
    d = 0.308

    J_c = 0.0107
    J_ei = -0.36
    J_ii = -0.12

    J_0 = 0.2112

    J_s = 0.4813

    J_se = 0.35
    J_ce = 0.0
    J_sp = 0.25

    J_es = -0.45

    J_pv = 0.45
    J_sv = -0.35
    J_vs = -0.35

    J_Aa = 3.8 * 0.8
    J_Ab = 1.75 * 0.8
    J_Ia = 3.2 * 0.8
    J_Ib = 3.2 * 0.8

    p1 = 1.0
    p2 = 1.0

    Iback_a = pp1  # 0.45
    Iback_b = pp1  # 0.45
    Iback_c = pp2  # 0.35

    Iback_d = 0.15
    Iback_e = 0.15

    Iback_f = 0.0
    Iback_g = 0.0

    I_a = 0.0
    I_b = 0.0
    I_c = 0.0

    I_va = 0.2
    I_vb = 0.2

    Q = [
        tau_a, tau_b, tau_c, tau_Aa, tau_Ab, tau_r, tau_n, tau_sa, tau_sb, tau_va, tau_vb,  # 11
        gamma_a, gamma_b, gamma_Aa, gamma_Ab, gamma_c, gamma_sa, gamma_sb, gamma_va, gamma_vb,  # 9
        gamma_ra, gamma_rb, gamma_rc, gamma_rs, gamma_v,  # 5
        g_I, c_1, c_0, r_0, a, b, d, J_c, J_ei, J_ii, J_0, J_s, J_se, J_ce, J_sp, J_es, J_pv, J_sv, J_vs, J_Aa, J_Ab, J_Ia, J_Ib, p1, p2, Iback_a, Iback_b, Iback_c, Iback_d, Iback_e, Iback_f, Iback_g, I_a, I_b, I_c, I_va, I_vb]

    # Input Parameters

    Qp0 = [np.array(Q)]
    Qp1 = [np.array(Q)]

    # Qp[0][50]=Iback_a
    # Qp[0][51]=Iback_b
    # Qp[0][52]=Iback_c

    # Qp[0][57]=I_a
    # Qp[0][58]=I_b
    # Qp[0][59]=I_c

    # Qp[0][60]=I_va
    # Qp[0][61]=I_vb
    ##

    Qp1[0][50] = pp1
    Qp1[0][51] = pp1
    Qp1[0][52] = pp2

    # this is where you can adjust the external input to the excitatory population
    Qp1[0][57] = 0.3
    Qp1[0][58] = 0.0
    Qp1[0][59] = 0.0

    # Qp1[0][60]=0.2
    # Qp1[0][61]=0.2

    # print([Qp1[0][57],Qp0[0][57]])

    # Solve the system
    z1 = [0.05, 0.05, 0.03, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.0,
          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    # Np=10
    # nn1=100001

    # Np=50
    # nn1=500001
    Np = 10
    nn1 = Np*1000+1

    Te = Np

    t = np.linspace(0, Np, nn1)
    T = np.linspace(0, Np*1000, nn1)

    clr = ['b', 'r']

    nn1-1
    gen = np.random.Generator(np.random.PCG64(123))
    from src.integral import itoint
    # d2 = sdeint.itoint(close(model_MW, *Qp1), close(g_fun, *P), z1, t, gen)
    d2 = itoint(close(model_MW, *Qp1), close(g_fun, *P), z1, t, gen)
    return


main()
