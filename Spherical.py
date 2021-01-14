import scipy as sp
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#spherical geometry correction: sec(Z) -> dl/dr
Re = float(6371000)
H = 7000
def cot(Z):
    return 1 / sp.tan(Z)

def x(r, Z):
    return sp.sqrt(((r + Re) ** 2 + Re ** 2 * (cot(Z) ** 2 / (1 + cot(Z) ** 2) - 1)) / (1 + cot(Z) ** 2)) - cot(Z) * Re / (1 + cot(Z) ** 2)

def dldr(r, Z):
    return (r + Re) * sp.sqrt(1 + cot(Z) ** 2) / (x(r, Z) * (1 + cot(Z) ** 2) + Re * cot(Z))

def midpoint_integrator(f, n_samp, lims, extraparams = None):
    deltax = (lims[1] - lims[0]) / n_samp
    x = sp.linspace(lims[0] + deltax / 2, lims[1] - deltax / 2, n_samp)
    if extraparams == None:
        samples = f(x)
    else:
        samples = f(x, extraparams)
    result = sp.sum(samples) * deltax
    return result

def x_(r, Z):
    return x(H * r, Z)

def integrand(r, Z):
    nonflip = sp.exp(-r) * (r * H + Re) / (x_(r, Z) * (1 + cot(Z) ** 2) + Re * cot(Z))
    flip = sp.exp(-1 / r) * (H / r + Re) / ((x_(1 / r, Z) * (1 + cot(Z) ** 2) + Re * cot(Z)) * r ** 2)
    return nonflip + flip
#%%

Z_ = sp.linspace(0.01, sp.pi / 2 - 0.01, 2000)
exponents = sp.zeros(2000)
for i in range(0, 2000):
    Z = Z_[i]
    exponents[i] = midpoint_integrator(integrand, 100000, [0, 1], extraparams = Z)
#%%
fig, ax = plt.subplots(1,2)
ax[1].plot(Z_ * 180 / sp.pi, exponents * sp.sqrt(1 + cot(Z_) ** 2), 'b-')
ax[1].plot(Z_ * 180 / sp.pi, exponents[0] * sp.sqrt(1 + cot(Z_[0]) ** 2) * sp.cos(Z_[0]) / sp.cos(Z_), 'r-')
ax[1].set_xlim(85,90)
ax[0].semilogy(Z_ * 180 / sp.pi, sp.exp(-exponents * sp.sqrt(1 + cot(Z_) ** 2)), 'b-')
ax[0].semilogy(Z_ * 180 / sp.pi, sp.exp(-exponents[0] * sp.sqrt(1 + cot(Z_[0]) ** 2) * sp.cos(Z_[0]) / sp.cos(Z_)),'r-')
fig.text(0.5, 0.04, 'Solar Zenith Angle ($\circ$)', ha='center', va='center', fontsize=13)
fig.show()

#%%
# Pressure dependence correction
P0 = 1e5
T0 = 280
M = 5.97e24
G = 6.67e-11
kB = 1.38e-23
m = 1.2 * kB * T0 / P0
def T(r):
    if r < 12000 + Re:
        return 283 - 75 * (r - Re) / 12e3
    elif r < 30000 + Re:
        return (r - Re - 12e3) * 5 / 6e3 + 208
    else:
        return 223
    
def dPdr(r, P, T):
    return -P * (2 / r + G * M * m / (r ** 2 * kB * T))

def P_profile(r, m, P0):
    return P0 * Re ** 2 * sp.exp(G * M * m * (1 / (r * kB * T(r)) - 1 / (Re * kB * T(Re)))) / r ** 2

def pressure_solver(step, P0, cutoff):
    r = [Re]
    P = [P0]
    while P[-1] / P[0] > cutoff:
        Pcurrent = P[-1]
        rcurrent = r[-1]
        P.append(Pcurrent + step * dPdr(rcurrent, Pcurrent, T(rcurrent)))
        r.append(rcurrent + step)
    return sp.array(r), sp.array(P)

def pressure_solverRK4(step, P0, cutoff):
    r = [Re]
    P = [P0]
    while P[-1] / P[0] > cutoff:
        Pcurrent = P[-1]
        rcurrent = r[-1]
        k1 = dPdr(rcurrent, Pcurrent, T(rcurrent))
        k2 = dPdr(rcurrent + step / 2, Pcurrent + step / 2 * k1, T(rcurrent + step / 2))
        k3 = dPdr(rcurrent + step / 2, Pcurrent + step / 2 * k2, T(rcurrent + step / 2))
        k4 = dPdr(rcurrent + step, Pcurrent + step * k3, T(rcurrent + step))
        P.append(Pcurrent + step / 6 * (k1 + 2 * k2 + 2 * k3 + k4))
        r.append(rcurrent + step)
    return sp.array(r), sp.array(P)

r, P = pressure_solver(100, P0, 1e-5)
P2 = sp.zeros(len(r))
for i in range(0, len(r)):    
    P2[i] = P_profile(r[i], m, P0)
n = sp.zeros(len(P))
for i in range(0, len(P)):
    n[i] = P[i] / (kB * T(r[i]))
plt.semilogy(r, n, 'b-')
plt.semilogy(r, n[0] * sp.exp(-(sp.array(r) - r[0]) / H), 'r-')
plt.show()
#%%
plt.plot(r, P, 'b-')
plt.plot(r, P2, 'r-')
plt.show()
#%%
def trapezium_rule(r, f):
    result = 0
    for i in range(0, len(r) - 1):
        result += (r[i + 1] - r[i]) * (f[i + 1] + f[i]) / 2
    return result

def path_integrator(step, P0, cutoff, Z):
    r, P = pressure_solverRK4(step, P0, cutoff)
    print(f'Range used is: {r[0]} - {r[-1]}.')
    n = sp.zeros(len(r))
    for i in range(0, len(r)):
        n[i] = P[i] / (kB * T(r[i]))
    exponents = sp.zeros(len(Z))
    for i in range(0, len(Z)):
        integrand = n * dldr(r - Re, Z[i])
        exponents[i] = trapezium_rule(r, integrand)
    return exponents

def exp_fit(r, H):
    return P0 * sp.exp(-(r - Re) / H)

fit, cov = curve_fit(exp_fit, sp.array(r), P, p0 = [7000])
plt.plot(r, P, 'b-')
plt.plot(r, exp_fit(sp.array(r), fit[0]), 'r-')
plt.show()
#%%
Z_ = sp.linspace(0.01, sp.pi / 2 - 0.01, 200)
exponents = path_integrator(1, 1e5, 1e-20, Z_)
exponents /= exponents[0]
adjusted = sp.exp(-exponents)
original = sp.exp(-exponents[0] * sp.cos(Z_[0]) / sp.cos(Z_))
#%%
plt.plot(Z_ * 180/sp.pi, adjusted*1000, color='cornflowerblue', ls='-', label='Geometric Correction')
plt.plot(Z_ * 180/sp.pi, original*1000, color='crimson', ls='-', label='Original Model')
plt.xlim(80, 90)
plt.ylim(-0.1, 0.0035*1000)
plt.ylabel('Intensity (Normalised)')
plt.xlabel('Solar Zenith Angle ($^\circ$)')
plt.grid()
plt.legend()
plt.show()
#%%
r, P1 = pressure_solverRK4(1, 1e5, 1e-10)
P2 = 1e5 * sp.exp(- (sp.array(r) - Re) / H)

error = (P2 - sp.array(P1)) / sp.array(P1)
plt.plot(r, error, 'kx')
plt.show()

print(trapezium_rule(r, P1) / trapezium_rule(r, P2))
#%%
r1, P1 = pressure_solver(1, 1e5, 1e-10)
r2, P2 = pressure_solverRK4(1, 1e5, 1e-10)
plt.plot(r1, P1, 'r-')
plt.plot(r2, P2, 'b-')
plt.show()
if len(r1) > len(r2):
    error = (sp.array(P1[:len(r2)]) - sp.array(P2)) / sp.array(P2)
    r = r2
elif len(r1) < len(r2):
    error = (sp.array(P1) - sp.array(P2[:len(r1)])) / sp.array(P2[:len(r1)])
    r = r1
else:
    error = (sp.array(P1) - sp.array(P2)) / sp.array(P2)
    r = r1
plt.plot(sp.array(r) / 1000, error, 'k-')
plt.show()
#%%

#Useful plots
params = {
   'axes.labelsize': 20,
   'font.size': 16,
   'xtick.labelsize': 15,
   'ytick.labelsize': 15,
   'figure.figsize': [8, 8/1.618]
   }
plt.rcParams.update(params)

# (1): error in atmospheric pressure / number density given and that predicted by spherically corrected DE
r, P1 = pressure_solverRK4(1, 1e5, 1e-10)
P2 = 1e5 * sp.exp(- (r - Re) / H)

Perror = (P2 - P1) / P1

fig, (ax1, ax2) = plt.subplots(2,1)

ax1.plot(r / 1000, Perror, color = 'k', linestyle='-', lw=2)
ax1.set_ylabel('Fractional Error in Pressure')
ax1.grid()

n1 = sp.zeros(len(r))
for i in range(len(r)):
    n1[i] = P1[i] / (kB * T(r[i]))
n2 = P2 / (kB * T(Re))
nerror = (n2 - n1) / n1

ax2.plot(r / 1000, nerror, 'k-')
ax2.set_ylabel('Fractional Error in Number Density')
ax2.grid()

fig.text(0.5, 0.04, 'Radius (km)', ha='center', va='center', fontsize=20)
fig.show()
#%%
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
# (2): sec(Z) dependence vs. full spherical corrections
# zero angle error:
thickness_correct = trapezium_rule(r, n1)
thickness_original = trapezium_rule(r, n2)
shift = thickness_original / thickness_correct
# angular dependence
Z = sp.linspace(0.01, sp.pi / 2 - 0.01, 200)
exponents = path_integrator(1, 1e5, 1e-10, Z)
exponents /= exponents[0]
adjusted = sp.exp(-exponents)
original = sp.exp(-exponents[0] * sp.cos(Z[0]) / sp.cos(Z))

fig, ax = plt.subplots()

ax.plot(Z * 180 / sp.pi, adjusted, color='cornflowerblue', ls='-', label = 'Spherically Corrected Dependence')
ax.plot(Z * 180 / sp.pi, original, color='crimson', ls='-', label = 'Flat Earth Approximation')
#ax.grid()
ax.legend()

axins = zoomed_inset_axes(ax, 2, loc='center', bbox_to_anchor=(0.36,0.30), bbox_transform=plt.gcf().transFigure)
axins.plot(Z * 180 / sp.pi, adjusted, color='cornflowerblue', ls='-')
axins.plot(Z * 180 / sp.pi, original, color='crimson', ls='-')
axins.grid()
axins.set_xticklabels([])
#axins.set_yticklabels([])
axins.set_xlim(70,90)
N = 0
while Z[N] < 70 * sp.pi / 180:
    N += 1
axins.set_ylim(-0.01, max(adjusted[N], original[N]) + 0.001)
mark_inset(ax, axins, loc1=3, loc2=1, fc='None', ec='0.5')
"""
ax2.plot(Z * 180 / sp.pi, adjusted, 'b-', label = 'Spherically corrected dependence')
ax2.plot(Z * 180 / sp.pi, original, 'r-', label = 'Flat Earth approximation')
ax2.grid()
ax2.legend()
ax2.set_title('High SZA Dependences')
ax2.set_xlim(70, 90)
N = 0
while Z[N] < 70 * sp.pi / 180:
    N += 1
ax2.set_ylim(0, max(adjusted[N], original[N]) + 0.001)
"""
fig.text(0.5, 0.04, 'Solar Zenith Angle ($^\circ$)', ha='center', va='center', fontsize=20)
fig.text(0.06, 0.5, 'Intensity (Normalised)', rotation=90, ha='center', va='center', fontsize=20)
fig.show()
#%%
# (3): error in sec(Z) approximation
experror = (original - adjusted) / adjusted
plt.plot(Z * 180 / sp.pi, experror, color='crimson', linestyle='-', linewidth=2)
plt.grid()
plt.xlabel('Solar Zenith Angle ($^\circ$)')
plt.ylabel('Fractional Error in Intensity')
plt.show()

#%%
#refractive index correction
dSun = 149.6e9
def humidity_profile(r, h0):
    #calculates partial water vapour pressure vs. height for given ground level humidity
    if r <= 8000 + Re:
        h = h0
    else:
        h = max(0, h0 - (r - Re - 8000) / 4000)
    P_sat = 10 ** (23.5518 - 2937.4 / T(r)) * T(r) ** (-4.9283) # in mbar/hPa
    P_partial = h * P_sat
    return P_partial

def refractive_index(r, P, h0): # Assuming lambda = 500nm
    ri = sp.zeros(len(r))
    for j in range(0, len(r)):
        ri[j] = 77.6 / T(r[j]) * (P[j] / 100 + 4810 * humidity_profile(r[j], h0) / T(r[j])) * 1e-6 + 1
    return ri

def thetasolver_singler(r_req, r, P, h0, k):
    if r_req <= r[-1]:
        N = 0
        while abs(r[N] - r_req) > abs(r[N + 1] - r_req):
            N += 1
        integrand = sp.sqrt(k / (refractive_index(r[:N + 1], P[:N + 1], h0) ** 2 * r[:N + 1] ** 4 - k * r[:N + 1] ** 2))
        theta = trapezium_rule(r[:N + 1], integrand)
    else:
        integrand1 = sp.sqrt(k / (refractive_index(r, P, h0) ** 2 * r ** 4 - k * r ** 2))
        theta_topatm = trapezium_rule(r, integrand1)
        if k != 0:
            M = int((r_req - r[-1]) * 10 / sp.sqrt(k))
        else:
            M = 1
        R = sp.linspace(r[-1], r_req, M + 1)
        integrand2 = sp.sqrt(k / (R ** 4 - k * R ** 2))
        theta = trapezium_rule(R, integrand2) + theta_topatm
    return theta

def thetasolver_multir(r, P, h0, k):
    integrand = sp.sqrt(k / (refractive_index(r, P, h0) ** 2 * r ** 4 - k * r ** 2))
    theta = sp.zeros(len(r))
    for i in range(1, len(r)):
        theta[i] = theta[i - 1] + (r[i] - r[i - 1]) * (integrand[i] + integrand[i - 1]) / 2
    return theta

def pathfinder(step, P0, cutoff, h0, Z, tolerance):
    r, P = pressure_solverRK4(step, P0, cutoff)
    k = int(2 * Z / sp.pi * Re ** 2)
    k2 = k
    kstep = Re ** 2 / 10
    start = thetasolver_singler(dSun, r, P, h0, k)
    if start > Z:
        k2 -= kstep
        while thetasolver_singler(dSun, r, P, h0, k2) > Z:
            k = k2
            k2 -= kstep
        bracket = [k2, k]
    else:
        k2 += kstep
        while thetasolver_singler(dSun, r, P, h0, k2) < Z:
            k = k2
            k2 += kstep
        bracket = [k, k2]
    finish = False
    while not finish:
        ktest = sp.mean(bracket)
        thetaSun = thetasolver_singler(dSun, r, P, h0, ktest)
        if abs(Z - thetaSun) < tolerance:
            finish = True
        if thetaSun > Z:
            bracket[1] = ktest
        else:
            bracket[0] = ktest
    return ktest, r, P

def pathfinder_interpolate(step, P0, cutoff, h0, Z, tolerance):
    r, P = pressure_solverRK4(step, P0, cutoff)
    k = int(2 * Z / sp.pi * Re ** 2)
    k2 = k
    kstep = Re ** 2 / 10
    start = thetasolver_singler(dSun, r, P, h0, k)
    if start > Z:
        k2 -= min(kstep, k)
        stop = thetasolver_singler(dSun, r, P, h0, k2)
        while stop > Z:
            k = k2
            start = stop
            k2 -= min(kstep, k)
            stop = thetasolver_singler(dSun, r, P, h0, k2)
        bracket = [k2, k]
        values = [stop, start]
    else:
        k2 += min(kstep, Re ** 2 - k)
        stop = thetasolver_singler(dSun, r, P, h0, k2)
        while stop < Z:
            k = k2
            start = stop
            k2 += min(kstep, Re ** 2 - k)
            stop = thetasolver_singler(dSun, r, P, h0, k2)
        bracket = [k, k2]
        values = [start, stop]
    finish = False
    while not finish:
        ktest = bracket[0] + (bracket[1] - bracket[0]) * (-(values[0] - Z) / (values[1] - values[0]))
        thetaSun = thetasolver_singler(dSun, r, P, h0, ktest)
        if abs(Z - thetaSun) < tolerance:
            finish = True
        if thetaSun > Z:
            bracket[1] = ktest
            values[1] = thetaSun
        else:
            bracket[0] = ktest
            values[0] = thetaSun
    return ktest, r, P

def curved_path_integrator(step, P0, cutoff, h0, Z, tolerance):
    k, r, P = pathfinder_interpolate(step, P0, cutoff, h0, Z, tolerance)
    theta = thetasolver_multir(r, P, h0, k)
    path = 0
    for i in range(0, len(r) - 1):
        dl = sp.sqrt((r[i + 1]- r[i]) ** 2 + sp.mean([r[i], r[i + 1]]) ** 2 * (theta[i + 1] - theta[i]) ** 2)
        n = (P[i] / (kB * T(r[i])) + P[i + 1] / (kB * T(r[i + 1]))) / 2
        path += dl * n
    return path
#%%

Z = sp.linspace(0.01, sp.pi / 2 - 0.01, 200)
exponents = path_integrator(1, 1e5, 1e-10, Z)
exponents2 = sp.zeros(len(Z))
exponents3 = sp.zeros(len(Z))
for i in range(0, len(Z)):
    exponents3[i] = midpoint_integrator(integrand, 100000, [0, 1], extraparams = Z[i])
    exponents2[i] = curved_path_integrator(1, 1e5, 1e-10, 0.5, Z[i], 0.001)
    print(i)
exponents /= exponents[0]
exponents2 /= exponents2[0]
exponents3 /= exponents3[0] * sp.sqrt(1 + cot(Z[0]) ** 2)
exponents4 = exponents[0] * sp.cos(Z[0]) / sp.cos(Z)
straight = sp.exp(-exponents)
curve = sp.exp(-exponents2)
simpleatmos = sp.exp(-exponents3 * sp.sqrt(1 + cot(Z) ** 2))
flatearth = sp.exp(-exponents4)
plt.semilogy(Z * 180 / sp.pi, straight, color='cornflowerblue', label = 'Spherical Correction')
plt.semilogy(Z * 180 / sp.pi, curve, color='fuchsia', label = 'Refraction and Spherical Correction')
plt.semilogy(Z * 180 / sp.pi, simpleatmos, color='springgreen', label = 'Geometric Approximation')
plt.semilogy(Z * 180 / sp.pi, flatearth, color='crimson', label = 'Original Approximation')
plt.grid()
plt.legend()
plt.xlim(80, 90)
#plt.ylim(0, 0.15)
plt.xlabel('Solar Zenith Angle ($^\circ$)')
plt.ylabel('Intensity (Normalised)')
plt.show()

#%%
"""
data = sp.array([Z, exponents, exponents2, exponents3, exponents4]).transpose()
sp.savetxt('Model_Comparison.txt', data)
"""
