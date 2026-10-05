r"""Ex_08.1 and Ex_08.2 (heat-sink version) — the physics: a power module on a
finned heat sink, steady and in a pulse of power.

*Deep Learning for Engineering* — MSc, Aalborg University.
Remus Teodorescu (ret@et.aau.dk), with support from Research Assistant
Noman Khan (nomank@energy.aau.dk).

**Reference texts.** Incropera & DeWitt, *Fundamentals of Heat and Mass
Transfer*, ch. 3 (fins, spreading) and ch. 5 (transient conduction); Liu,
*PINN with Python: An Introduction* (2025). The code, the problem and the
exposition here are original to this course.

## The problem

A power module sits on the base plate of an extruded aluminium heat sink and
dissipates a known power over its footprint. The heat spreads through the base
plate and leaves through the fins into forced air. The cross-section is solved:
``x`` across the base (0 to W), ``z`` through its thickness (0 at the fin side,
``t_b`` at the module side), per metre of depth.

    rho c_p T_t = k (T_xx + T_zz)             in the base plate
    k T_z = q''(x) s(t)                         at z = t_b   (the module's flux, on its footprint)
    k T_z = h_eff (T - T_inf)                   at z = 0     (the fins, collapsed into one coefficient)
    T_x = 0                                     at x = 0, W  (the ends of the base are insulated)
    T = T_inf                                   at t = 0     (Ex_08.2: the sink starts at the air temperature)

**The fins enter exactly.** A straight fin of thickness ``t_f`` and length
``L_f`` with the coefficient ``h`` on both faces has the textbook efficiency
``eta = tanh(m L_c) / (m L_c)``, ``m = sqrt(2h / (k t_f))``, ``L_c = L_f + t_f/2``.
Per pitch ``p`` of base the fins and the bare base between them take
``h (eta * 2 L_c + (p - t_f))`` watts per kelvin and metre of depth, so the
finned face acts on the base plate as one coefficient
``h_eff = h (eta * 2 L_c + p - t_f) / p``. That is the standard heat-sink
calculation, and it is what makes the base plate a rectangle whose solution
is exact.

## The exact solutions

Insulated ends make cosines in ``x`` exact. Steady, the field is

    theta(x, z) = sum_n cos(lam_n x) Z_n(z),     lam_n = n pi / W

with ``Z_0`` linear and every other ``Z_n`` a combination of ``cosh`` and
``sinh`` fixed by the two faces: elementary functions, no eigenvalue to find
(:func:`exact`). In time, what has not yet arrived is expanded in the
eigenfunctions ``cos(mu_j (t_b - z))`` of the Robin face, ``mu tan(mu t_b) =
h_eff / k`` (the transcendental equation of Incropera ch. 5), and every mode
dies as ``exp(-alpha (lam_n^2 + mu_j^2) t)`` (:func:`exact_transient`). A
pulse is two steps, one subtracted from the other.

## Scaled units

``X = x / W``, ``Z = z / t_b``, and the rise above the air over
``THETA = q_mean / h_eff``, the rise a one-dimensional hand calculation gives
the fin side. In those units the base plate is the unit square and

    EPS^2 theta_XX + theta_ZZ = 0                       EPS = t_b / W
    theta_Z = BI g(X) s(tau)       at Z = 1              g = q'' / q_mean on the footprint, 0 elsewhere
    theta_Z = BI theta             at Z = 0              BI = h_eff t_b / k
    theta_X = 0                    at X = 0, 1

and in time ``theta_tau = FO (EPS^2 theta_XX + theta_ZZ)`` with
``FO = alpha t_end / t_b^2``. Everything in this module works in the scaled
units; :func:`kelvin` converts a rise back.

Values (C10 rule 5), typical and ASSUMED until cited: aluminium 6061
(167 W/m K, 2.43 MJ/m^3 K); a 100 mm wide, 10 mm thick base with ten fins of
2 x 40 mm at a 10 mm pitch; forced air at 40 degC with h = 40 W/m^2 K; a
module of 150 W on a 30 mm wide footprint, per 100 mm of depth.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import brentq

__all__ = [
    "W_BASE", "T_BASE", "K_AL", "RHO_CP", "ALPHA", "N_FINS", "T_FIN", "L_FIN", "PITCH",
    "H_AIR", "T_AIR", "DEPTH", "POWER", "MODULE", "FLUX", "Q_MEAN",
    "M_FIN", "ETA_FIN", "H_EFF", "THETA", "BI", "EPS", "TAU_LUMPED",
    "T_ON", "T_END", "FO", "SENSOR", "NOISE_K", "DT_SENSOR",
    "kelvin", "footprint", "fin_efficiency", "exact", "exact_transient", "roots",
    "fdm_steady", "fdm_transient", "hottest",
    "sample_base", "sample_faces", "sample_base_time", "sample_faces_time",
    "pulse", "sensor_readings", "draw_heatsink", "describe_problem",
]

# ------------------------------------------------------------------ the data
W_BASE = 0.100           #: m, the base plate across the fins
T_BASE = 0.010           #: m, its thickness
K_AL = 167.0             #: W/(m·K), aluminium alloy 6061
RHO_CP = 2.43e6          #: J/(m³K)
ALPHA = K_AL / RHO_CP    #: m²/s, the diffusivity, 6.87e-05
N_FINS = 10              #: straight fins along the base
T_FIN = 0.002            #: m, fin thickness
L_FIN = 0.040            #: m, fin length
PITCH = W_BASE / N_FINS  #: m, fin pitch, 10 mm
H_AIR = 40.0             #: W/(m²K), forced air on the fins and the bare base   [typical value]
T_AIR = 40.0             #: degC, the air in the enclosure
DEPTH = 0.100            #: m, the heat sink into the page; the module covers the whole depth
POWER = 150.0            #: W, the module's losses                                  [typical value]
MODULE = (0.035, 0.065)  #: m, the module's footprint across the base: 30 mm, centred
FLUX = POWER / ((MODULE[1] - MODULE[0]) * DEPTH)   #: W/m², 50 kW/m² on the footprint
Q_MEAN = POWER / (W_BASE * DEPTH)                  #: W/m², the same heat spread over the whole base, 15 kW/m²


def fin_efficiency(h=H_AIR):
    """The straight-fin efficiency tanh(mL_c)/(mL_c) and the effective
    coefficient the finned face puts on the base plate, for a coefficient h."""
    m = np.sqrt(2.0 * h / (K_AL * T_FIN))
    lc = L_FIN + T_FIN / 2
    eta = np.tanh(m * lc) / (m * lc)
    h_eff = h * (eta * 2 * lc + PITCH - T_FIN) / PITCH
    return m, eta, h_eff


M_FIN, ETA_FIN, H_EFF = fin_efficiency()
THETA = Q_MEAN / H_EFF          #: K, the temperature scale: the fin side by a 1-D hand calculation
BI = H_EFF * T_BASE / K_AL      #: the Biot number of the base plate
EPS = T_BASE / W_BASE           #: the aspect ratio: thickness over width
TAU_LUMPED = RHO_CP * T_BASE / H_EFF   #: s, the lumped time constant of the base

# the pulse (Ex_08.2)
T_ON = 20.0                     #: s, the module's power is on from 0 to T_ON
T_END = 150.0                   #: s, the window
FO = ALPHA * T_END / T_BASE ** 2   #: the Fourier number of the window on the thickness
SENSOR = 0.080                  #: m, a thermocouple glued to the module face of the base, beside the module
NOISE_K = 0.05                  #: K, its noise
DT_SENSOR = 2.0                 #: s, its sampling interval


def kelvin(theta):
    """A scaled rise back to kelvin above the air."""
    return THETA * np.asarray(theta)


def footprint(X):
    """g(X) = q''/q_mean on the footprint, 0 elsewhere (scaled x)."""
    X = np.asarray(X)
    lo, hi = MODULE[0] / W_BASE, MODULE[1] / W_BASE
    return np.where((X >= lo) & (X <= hi), FLUX / Q_MEAN, 0.0)


def pulse(tau):
    """s(tau): 1 while the module is on, 0 after (scaled time)."""
    return np.where(np.asarray(tau) <= T_ON / T_END, 1.0, 0.0)


# ------------------------------------------------------------ the exact field
def _coeff(lo, hi, n):
    """Cosine coefficients of the indicator of (lo, hi) on (0, W)."""
    m = np.arange(n)
    c = np.empty(n)
    c[0] = (hi - lo) / W_BASE
    c[1:] = 2.0 / (m[1:] * np.pi) * (np.sin(m[1:] * np.pi * hi / W_BASE) - np.sin(m[1:] * np.pi * lo / W_BASE))
    return c


def _modes(n, patches=((MODULE[0], MODULE[1], FLUX),)):
    """q_n, the cosine coefficients of the module's flux, and lam_n."""
    q = np.zeros(n)
    for lo, hi, flux in patches:
        q += flux * _coeff(lo, hi, n)
    lam = np.arange(n) * np.pi / W_BASE
    return q, lam


def _Z_steady(q, lam, z):
    """Z_n(z) for every mode at the heights z (array): the steady profile
    through the thickness, in kelvin. Hyperbolic functions written with
    decaying exponentials so nothing overflows."""
    z = np.asarray(z, dtype=float)
    Z = np.empty((len(q), z.size))
    Z[0] = q[0] * (z / K_AL + 1.0 / H_EFF)
    l = lam[1:, None]
    r = H_EFF / (K_AL * l)
    num = (1 + r) * np.exp(l * (z - T_BASE)) + (1 - r) * np.exp(-l * (z + T_BASE))
    den = (1 + r) - (1 - r) * np.exp(-2 * l * T_BASE)
    Z[1:] = q[1:, None] / (K_AL * l) * num / den
    return Z


def _dilog(w):
    """Li_2(w) for complex w, |w| <= 1: scipy's spence(z) is Li_2(1 - z)."""
    from scipy.special import spence
    return spence(1.0 - w)


def _flat_tail(x, z, lo, hi, flux):
    """sum_{n>=1} q_n cos(lam_n x) e^{-lam_n (t_b - z)} / (k lam_n), in closed
    form: the series every mode tends to once lam_n t_b is large. Its slow
    1/n convergence at the footprint's edges is what the closed form removes.
    With phi = pi x / W etc., sum e^{-n eps} sin(n a) cos(n b) / n^2 is the
    imaginary part of a dilogarithm."""
    s = np.pi / W_BASE
    eps = s * (T_BASE - z)
    out = 0.0
    for sign, edge in ((1.0, hi), (-1.0, lo)):
        for pm in (1.0, -1.0):
            out = out + sign * np.imag(_dilog(np.exp(-eps + 1j * s * (edge + pm * x))))
    return flux * W_BASE / (K_AL * np.pi ** 2) * out


def exact(x, z, n=200, patches=None):
    """The steady rise in kelvin at the points (x, z) in metres, arrays of the
    same shape. Exact to rounding: the mean term, the slowly converging part
    of the series in closed form, and a remainder that converges like
    exp(-2 lam_n t_b), summed to n terms."""
    x = np.asarray(x, dtype=float); z = np.asarray(z, dtype=float)
    shape = np.broadcast(x, z).shape
    x = np.broadcast_to(x, shape).ravel(); z = np.broadcast_to(z, shape).ravel()
    if patches is None:
        patches = ((MODULE[0], MODULE[1], FLUX),)
    q, lam = _modes(n, patches)
    zs, inv = np.unique(z, return_inverse=True)
    Z = _Z_steady(q, lam, zs)                      # (n, nz)
    l = lam[1:, None]
    Z[1:] -= q[1:, None] / (K_AL * l) * np.exp(-l * (T_BASE - zs[None, :]))   # take the flat tail out of every mode
    Z = Z[:, inv]
    C = np.cos(np.outer(lam, x))
    th = np.sum(C * Z, axis=0)
    for lo, hi, flux in patches:
        th = th + _flat_tail(x, z, lo, hi, flux)
    return th.reshape(shape)


def roots(k):
    """The first k roots mu_j of mu tan(mu t_b) = h_eff / k_al."""
    beta = H_EFF / K_AL
    f = lambda mu: mu * np.sin(mu * T_BASE) - beta * np.cos(mu * T_BASE)
    return np.array([brentq(f, j * np.pi / T_BASE + 1e-9, (j + 0.5) * np.pi / T_BASE - 1e-9) for j in range(k)])


def _step(x, z, t, n, k, mu):
    """The rise in kelvin at time t after the module is switched on, from a
    sink at the air temperature: the steady field less what has not arrived."""
    x = np.asarray(x, dtype=float); z = np.asarray(z, dtype=float)
    q, lam = _modes(n)
    norm = T_BASE / 2 + np.sin(2 * mu * T_BASE) / (4 * mu)
    psi = np.cos(np.outer(mu, T_BASE - z))        # (k, N)
    s = lam[:, None] ** 2 + mu[None, :] ** 2       # (n, k)
    A = (q[:, None] / K_AL / s / norm[None, :]) * np.exp(-ALPHA * s * t)   # (n, k)
    C = np.cos(np.outer(lam, x))                   # (n, N)
    return exact(x, z, n) - np.einsum("nk,kN,nN->N", A, psi, C)


def exact_transient(x, z, t, n=300, k=120):
    """The rise in kelvin at the points (x, z) in metres and the time t in
    seconds (a scalar), for the pulse: on at 0, off at T_ON. Two steps, the
    second subtracted. Pass t_on=None for the step alone."""
    x = np.asarray(x, dtype=float); z = np.asarray(z, dtype=float)
    shape = np.broadcast(x, z).shape
    xf = np.broadcast_to(x, shape).ravel(); zf = np.broadcast_to(z, shape).ravel()
    if t <= 0:
        return np.zeros(shape)
    mu = roots(k)
    th = _step(xf, zf, t, n, k, mu)
    if t > T_ON:
        th = th - _step(xf, zf, t - T_ON, n, k, mu)
    return th.reshape(shape)


def hottest(n=400):
    """The hottest point of the steady field: on the module face, under the
    centre of the footprint, in kelvin above the air."""
    return float(exact(np.array([W_BASE / 2]), np.array([T_BASE]), n)[0])


# ----------------------------------------------------------- finite differences
def _operator(nx, nz):
    """The scaled operator EPS^2 d_XX + d_ZZ on an nx by nz grid of the unit
    square, with the insulated ends and the two faces built in by ghost
    nodes; returns the matrix and the right-hand-side vector of the flux."""
    hx, hz = 1.0 / (nx - 1), 1.0 / (nz - 1)
    Dx = sp.diags([1.0, -2.0, 1.0], [-1, 0, 1], shape=(nx, nx)).tolil()
    Dx[0, 1] = 2.0; Dx[-1, -2] = 2.0                         # mirrored nodes: theta_X = 0 at both ends
    Dz = sp.diags([1.0, -2.0, 1.0], [-1, 0, 1], shape=(nz, nz)).tolil()
    Dz[0, 1] = 2.0; Dz[0, 0] = -2.0 - 2.0 * hz * BI           # bottom ghost: theta_Z = BI theta
    Dz[-1, -2] = 2.0                                          # top ghost: theta_Z = BI g, the flux goes to the vector
    A = (EPS ** 2 / hx ** 2) * sp.kron(sp.identity(nz), Dx.tocsr()) + (1.0 / hz ** 2) * sp.kron(Dz.tocsr(), sp.identity(nx))
    X = np.linspace(0, 1, nx)
    g = footprint(X)
    edge = np.isclose(X, MODULE[0] / W_BASE) | np.isclose(X, MODULE[1] / W_BASE)
    g = np.where(edge, 0.5 * FLUX / Q_MEAN, g)                # a node on the footprint's edge carries half the flux
    b = np.zeros(nx * nz)
    b[-nx:] = 2.0 * BI * g / hz                               # the flux enters the top row through its ghost node
    return A.tocsc(), b, X, np.linspace(0, 1, nz)


def fdm_steady(nx, nz):
    """The steady scaled rise on an nx by nz grid: X, Z (1-D) and theta (nz, nx)."""
    A, b, X, Z = _operator(nx, nz)
    th = spla.spsolve(A, -b)
    return X, Z, th.reshape(nz, nx)


def fdm_transient(nx, nz, nt, record):
    """Crank-Nicolson on an nx by nz grid with nt steps over the window: the
    scaled rise at the recorded scaled times (a sorted array, multiples of
    1/nt), as (len(record), nz, nx). The pulse switches off at the nearest
    step; T_ON / T_END times nt should be an integer."""
    A, b, X, Z = _operator(nx, nz)
    dt = 1.0 / nt
    I = sp.identity(nx * nz, format="csc")
    L = spla.splu((I - 0.5 * dt * FO * A).tocsc())
    R = (I + 0.5 * dt * FO * A).tocsc()
    th = np.zeros(nx * nz)
    out, rec = [], list(np.round(np.asarray(record) * nt).astype(int))
    if rec and rec[0] == 0:
        out.append(th.copy().reshape(nz, nx)); rec = rec[1:]
    for step in range(1, nt + 1):
        s0, s1 = pulse((step - 1) * dt), pulse(step * dt)
        th = L.solve(R @ th + 0.5 * dt * FO * (s0 + s1) * b)
        if rec and step == rec[0]:
            out.append(th.copy().reshape(nz, nx)); rec = rec[1:]
    return X, Z, np.array(out)


# ------------------------------------------------------------------- sampling
def sample_base(n, seed=88):
    """n points in the unit square (X, Z), Latin hypercube."""
    rng = np.random.default_rng(seed)
    u = (rng.permuted(np.tile(np.arange(n), (2, 1)), axis=1).T + rng.random((n, 2))) / n
    return u


def sample_faces(n_face, n_end, seed=88):
    """Points on the four sides of the unit square: (top, bottom, left, right),
    each an (m, 2) array of (X, Z). The top is sampled more densely round the
    footprint's edges, where the flux steps."""
    rng = np.random.default_rng(seed)
    lo, hi = MODULE[0] / W_BASE, MODULE[1] / W_BASE
    xt = np.concatenate([rng.random(n_face), lo + 0.06 * (rng.random(n_face // 4) - 0.5), hi + 0.06 * (rng.random(n_face // 4) - 0.5)])
    xt = np.clip(xt, 0, 1)
    top = np.c_[xt, np.ones_like(xt)]
    xb = rng.random(n_face); bottom = np.c_[xb, np.zeros_like(xb)]
    zl = rng.random(n_end); left = np.c_[np.zeros_like(zl), zl]
    zr = rng.random(n_end); right = np.c_[np.ones_like(zr), zr]
    return top, bottom, left, right


def sample_base_time(n, seed=88):
    """n points (X, Z, tau) in the unit cube; half of them in the first fifth
    of the window, where the field changes fastest."""
    rng = np.random.default_rng(seed)
    u = (rng.permuted(np.tile(np.arange(n), (3, 1)), axis=1).T + rng.random((n, 3))) / n
    half = n // 2
    u[:half, 2] *= 0.2
    return u


def sample_faces_time(n_face, n_end, n_t, seed=88):
    """The four faces at random instants: (top, bottom, left, right), each (m, 3)."""
    rng = np.random.default_rng(seed + 1)
    top, bottom, left, right = sample_faces(n_face, n_end, seed)
    def with_time(p):
        tau = rng.random(len(p)); tau[: len(p) // 2] *= 0.2
        return np.c_[p, tau]
    return with_time(top), with_time(bottom), with_time(left), with_time(right)


def sensor_readings(seed=88):
    """The thermocouple's record over the window: the times in seconds and the
    rise in kelvin above the air at (SENSOR, T_BASE), every DT_SENSOR seconds,
    with NOISE_K of Gaussian noise. From the exact solution."""
    rng = np.random.default_rng(seed + 7)
    t = np.arange(DT_SENSOR, T_END + 1e-9, DT_SENSOR)
    th = np.array([exact_transient(np.array([SENSOR]), np.array([T_BASE]), tt)[0] for tt in t])
    return t, th + NOISE_K * rng.standard_normal(len(t))


# ------------------------------------------------------------------ drawing
def draw_heatsink(ax, show_sensor=False, show_domain=True):
    """The cross-section, in millimetres: base plate, fins, module, air."""
    import matplotlib.patches as mpatches
    mm = 1e3
    W, tb, Lf, tf, p = W_BASE * mm, T_BASE * mm, L_FIN * mm, T_FIN * mm, PITCH * mm
    ax.add_patch(mpatches.Rectangle((0, 0), W, tb, facecolor="0.80", edgecolor="k", lw=1.2))
    for i in range(N_FINS):
        xc = (i + 0.5) * p
        ax.add_patch(mpatches.Rectangle((xc - tf / 2, -Lf), tf, Lf, facecolor="0.80", edgecolor="k", lw=0.8))
    ax.add_patch(mpatches.Rectangle((MODULE[0] * mm, tb), (MODULE[1] - MODULE[0]) * mm, 6, facecolor="tab:red", alpha=0.35, edgecolor="tab:red", lw=1.2))
    ax.text((MODULE[0] + MODULE[1]) / 2 * mm, tb + 3, f"module, {POWER:.0f} W", ha="center", va="center", fontsize=9, color="tab:red")
    for xa in np.linspace(8, W - 8, 6):
        ax.annotate("", xy=(xa, -Lf - 2), xytext=(xa, -Lf - 10), arrowprops=dict(arrowstyle="->", color="tab:blue", lw=1))
    ax.text(W / 2, -Lf - 13, f"forced air, {T_AIR:.0f} °C, h = {H_AIR:.0f} W/(m²K)", ha="center", va="top", fontsize=9, color="tab:blue")
    if show_domain:
        ax.add_patch(mpatches.Rectangle((0, 0), W, tb, fill=False, edgecolor="tab:orange", lw=2, ls="--"))
        ax.text(W + 2, tb / 2, "the base plate:\nthe domain solved", va="center", fontsize=8.5, color="tab:orange")
        ax.text(W + 2, -Lf / 2, f"the fins: η = {ETA_FIN:.3f},\nh_eff = {H_EFF:.0f} W/(m²K)\non the fin side", va="center", fontsize=8.5, color="0.3")
    if show_sensor:
        ax.plot([SENSOR * mm], [tb], "kv", ms=9, zorder=5)
        ax.text(SENSOR * mm, tb + 2, "thermocouple", ha="center", va="bottom", fontsize=8.5)
    ax.annotate("", xy=(0, tb + 11), xytext=(W, tb + 11), arrowprops=dict(arrowstyle="<->", lw=0.8))
    ax.text(W / 2, tb + 12, f"{W:.0f} mm", ha="center", va="bottom", fontsize=8.5)
    ax.annotate("", xy=(-4, 0), xytext=(-4, tb), arrowprops=dict(arrowstyle="<->", lw=0.8))
    ax.text(-6, tb / 2, f"{tb:.0f} mm", ha="right", va="center", fontsize=8.5)
    ax.annotate("", xy=(-4, -Lf), xytext=(-4, 0), arrowprops=dict(arrowstyle="<->", lw=0.8))
    ax.text(-6, -Lf / 2, f"{Lf:.0f} mm", ha="right", va="center", fontsize=8.5)
    ax.set_xlim(-22, W + 40); ax.set_ylim(-Lf - 20, tb + 18)
    ax.set_aspect("equal"); ax.set_xlabel("x  [mm]"); ax.set_ylabel("z  [mm]")
    ax.set_title("the heat sink in cross-section, per metre of depth", fontsize=10)


def describe_problem():
    print(f"  base plate : {W_BASE * 1e3:.0f} x {T_BASE * 1e3:.0f} mm aluminium, k = {K_AL:.0f} W/(m·K), rho c_p = {RHO_CP / 1e6:.2f} MJ/(m³K)")
    print(f"  fins       : {N_FINS} of {T_FIN * 1e3:.0f} x {L_FIN * 1e3:.0f} mm at a {PITCH * 1e3:.0f} mm pitch, in air at {T_AIR:.0f} °C with h = {H_AIR:.0f} W/(m²K)")
    print(f"  fin        : m = {M_FIN:.2f} /m, m L_c = {M_FIN * (L_FIN + T_FIN / 2):.3f}, efficiency {ETA_FIN:.3f}; "
          f"the finned face acts as h_eff = {H_EFF:.1f} W/(m²K), {H_EFF / H_AIR:.1f} times the bare h")
    print(f"  module     : {POWER:.0f} W on {(MODULE[1] - MODULE[0]) * 1e3:.0f} x {DEPTH * 1e3:.0f} mm: {FLUX / 1e3:.0f} kW/m² on the footprint, "
          f"{Q_MEAN / 1e3:.0f} kW/m² over the whole base")
    print(f"  scales     : a 1-D hand calculation puts the fin side {THETA:.2f} K above the air and adds "
          f"q_mean t_b / k = {Q_MEAN * T_BASE / K_AL:.2f} K through the thickness; Bi = {BI:.4f}; "
          f"lumped time constant {TAU_LUMPED:.1f} s")
