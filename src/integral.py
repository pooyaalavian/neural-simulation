import numpy as np 
from sdeint.integrate import _check_args, deltaW, Ikpw

def itoint(f, G, y0, tspan, generator=None):
    (d, m, f, G, y0, tspan, __, __) = _check_args(f, G, y0, tspan, None, None)
    chosenAlgorithm = itoSRI2
    return chosenAlgorithm(f, G, y0, tspan, generator=generator)



def itoSRI2(f, G, y0, tspan, Imethod=Ikpw, dW=None, I=None, generator=None):
    return _Roessler2010_SRK2(f, G, y0, tspan, Imethod, dW, I, generator)

def _Roessler2010_SRK2(f, G, y0, tspan, IJmethod, dW=None, IJ=None,
                       generator=None):
    """Implements the Roessler2010 order 1.0 strong Stochastic Runge-Kutta
    algorithms SRI2 (for Ito equations) and SRS2 (for Stratonovich equations).

    Algorithms SRI2 and SRS2 are almost identical and have the same extended
    Butcher tableaus. The difference is that Ito repeateded integrals I_ij are
    replaced by Stratonovich repeated integrals J_ij when integrating a
    Stratonovich equation (Theorem 6.2 in Roessler2010).

    Args:
      f: A function f(y, t) returning an array of shape (d,)
      G: Either a function G(y, t) that returns an array of shape (d, m),
         or a list of m functions g(y, t) each returning an array shape (d,).
      y0: array of shape (d,) giving the initial state
      tspan (array): Sequence of equally spaced time points
      IJmethod (callable): which function to use to generate repeated
        integrals. N.B. for an Ito equation, must use an Ito version here
        (either Ikpw or Iwik). For a Stratonovich equation, must use a
        Stratonovich version here (Jkpw or Jwik).
      dW: optional array of shape (len(tspan)-1, d).
      IJ: optional array of shape (len(tspan)-1, m, m).
        Optional arguments dW and IJ are for advanced use, if you want to
        use a specific realization of the d independent Wiener processes and
        their multiple integrals at each time step. If not provided, suitable
        values will be generated randomly.
      generator (numpy.random.Generator, optional) Random number generator
        instance to use. If omitted, a new default_rng will be instantiated.

    Returns:
      y: array, with shape (len(tspan), len(y0))

    Raises:
      SDEValueError

    See also:
      A. Roessler (2010) Runge-Kutta Methods for the Strong Approximation of
        Solutions of Stochastic Differential Equations
    """
    (d, m, f, G, y0, tspan, dW, IJ) = _check_args(f, G, y0, tspan, dW, IJ)
    if generator is None and (dW is None or IJ is None):
        generator = np.random.default_rng()
    have_separate_g = (not callable(G)) # if G is given as m separate functions
    N = len(tspan)
    h = (tspan[N-1] - tspan[0])/(N - 1) # assuming equal time steps
    if dW is None:
        # pre-generate Wiener increments (for m independent Wiener processes):
        dW = deltaW(N - 1, m, h, generator) # shape (N, m)
    if IJ is None:
        # pre-generate repeated stochastic integrals for each time step.
        # Must give I_ij for the Ito case or J_ij for the Stratonovich case:
        __, I = IJmethod(dW, h, generator=generator) # shape (N, m, m)
    else:
        I = IJ
    # allocate space for result
    y = np.zeros((N, d), dtype=y0.dtype)
    y[0] = y0
    Gn = np.zeros((d, m), dtype=y.dtype)
    for n in range(0, N-1):
        tn = tspan[n]
        tn1 = tspan[n+1]
        h = tn1 - tn
        sqrth = np.sqrt(h)
        Yn = y[n] # shape (d,)
        Ik = dW[n,:] # shape (m,)
        Iij = I[n,:,:] # shape (m, m)
        fnh = f(Yn, tn)*h # shape (d,)
        if have_separate_g:
            for k in range(0, m):
                Gn[:,k] = G[k](Yn, tn)
        else:
            Gn = G(Yn, tn)
        sum1 = np.dot(Gn, Iij)/sqrth # shape (d, m)
        H20 = Yn + fnh # shape (d,)
        H20b = np.reshape(H20, (d, 1))
        H2 = H20b + sum1 # shape (d, m)
        H30 = Yn
        H3 = H20b - sum1
        fn1h = f(H20, tn1)*h
        dg = np.dot(Gn, Ik)
        Yn1 = Yn + 0.5*(fnh + fn1h) + dg
        print(tn, dg)
        # if have_separate_g:
        #     for k in range(0, m):
        #         Yn1 += 0.5*sqrth*(G[k](H2[:,k], tn1) - G[k](H3[:,k], tn1))
        # else:
        #     for k in range(0, m):
        #         Yn1 += 0.5*sqrth*(G(H2[:,k], tn1)[:,k] - G(H3[:,k], tn1)[:,k])
        y[n+1] = Yn1
    return y

