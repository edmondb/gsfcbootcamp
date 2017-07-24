import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def lorenz(X, t, sigma, beta, rho):
    """The Lorenz equations."""
    u, v, w = X
    up = -sigma*(u - v)
    vp = rho*u - v - u*w
    wp = -beta*w + u*v
    return up, vp, wp

# Lorenz paramters and initial conditions
sigma, beta, rho = 10, 2.667, 28
u0, v0, w0 = 0, 1, 1.05

# Maximum time point and total number of time points
tmax, n = 100, 10000

# Create an array of 10000 time values such that t = [0,80]
t = np.linspace(0, tmax, n)


# Integrate the Lorenz equations on the time grid t
f = odeint(lorenz, (u0, v0, w0), t, args=(sigma, beta, rho))
x, y, z = f.T

# Plot the Lorenz attractor using a Matplotlib 3D projection
fig, ax = plt.subplots()

# Plot x vs z (red-dashed line, use linewidth=0.5)
ax.plot(x, z, 'r--', linewidth=0.5)

# Add title and x/y labels (fontsize=18)
ax.set_title('Lorenz attractor')
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')

# Remove all the axis clutter, leaving just the curve.
#ax.set_axis_off()

# Optional: Repeat plot for x vs y, y vs z - i.e. plot all three in a 1x3 sunplot
fig, axes = plt.subplots(1, 3)
axes[0].plot(x, z, 'r--', linewidth=0.5)
axes[1].plot(x, y, 'r--', linewidth=0.5)
axes[2].plot(y, z, 'r--', linewidth=0.5)

plt.show()
