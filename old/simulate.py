import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# Parameters
alpha = 1.0  # Convection speed
r0 = np.array([0.1, 0.1])  # Point to which density converges
grid_size = 100  # Grid resolution
dt = 0.01  # Time step
total_time = 3  # Total simulation time
time_steps = int(total_time / dt)

# Create grid
x = np.linspace(0, 1, grid_size)
y = np.linspace(0, 1, grid_size)
X, Y = np.meshgrid(x, y)
dx = x[1] - x[0]

# Initialize density distribution (e.g., Gaussian distribution)
def initial_density(X, Y, mean, std):
    return np.exp(-(((X - mean[0])**2 + (Y - mean[1])**2) / (2 * std**2))) / (2 * np.pi * std**2)

# Specify mean and standard deviation
mean = [0.5, 0.5]  # Mean of the Gaussian
std = 0.05  # Standard deviation of the Gaussian

p = initial_density(X, Y, mean, std)

# Velocity field towards r0
def velocity_field(X, Y, r0):
    vx = alpha * (r0[0] - X)
    vy = alpha * (r0[1] - Y)
    return vx, vy

vx, vy = velocity_field(X, Y, r0)

# Finite difference scheme for convection
def update_density(p, vx, vy, dt, dx):

    
    # Upwind scheme for x-direction
    dpdt_x = np.where(vx > 0,
                      -vx * (p - np.roll(p, 1, axis=1)) / dx,  # Backward difference
                      -vx * (np.roll(p, -1, axis=1) - p) / dx) # Forward difference

    # Upwind scheme for y-direction
    dpdt_y = np.where(vy > 0,
                      -vy * (p - np.roll(p, 1, axis=0)) / dx,  # Backward difference
                      -vy * (np.roll(p, -1, axis=0) - p) / dx) # Forward difference
    

    # Central difference scheme
    dpdt_x = -vx * (np.roll(p, -1, axis=1) - np.roll(p, 1, axis=1)) / (2 * dx)
    dpdt_y = -vy * (np.roll(p, -1, axis=0) - np.roll(p, 1, axis=0)) / (2 * dx)
    dpdt = dpdt_x + dpdt_y + 2 * p

    # Set boundary conditions to have 0 derivative
    dpdt[:, 0] = 0
    dpdt[:, -1] = 0
    dpdt[0, :] = 0
    dpdt[-1, :] = 0
    return p + dt * dpdt

hist_p = [p.copy()]

# Visualization setup
fig, ax = plt.subplots()
contour = ax.contourf(X, Y, p, levels=50, cmap='viridis')
plt.colorbar(contour)

# Update function for animation
def update(frame):
    global p
    p = update_density(p, vx, vy, dt, dx)
    hist_p.append(p.copy())
    for c in ax.collections:
        c.remove()
    p[p<0] = 0
    ax.contourf(X, Y, p, levels=50, cmap='viridis')
    return ax.collections


# Create animation
ani = FuncAnimation(fig, update, frames=time_steps, interval=50, blit=False)
ani.save('density_convection.gif', writer=PillowWriter(fps=30))

# plt.show()

# Assuming hist_p contains the history of p for all frames
last_frame = hist_p[-1]

# Plot the last frame
# plt.clf()
# plt.figure()
# plt.contourf(X, Y, last_frame, levels=50, cmap='viridis')
# plt.colorbar()  # To show the color scale
# plt.title('Last Frame of Density Convection')
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.show()


# Plot density
plt.clf()
hist_t = [dt * i for i in range(len(hist_p))]
hist_p_sum = [np.sum(p) * dx**2 for p in hist_p]
plt.plot(hist_t, hist_p_sum)
plt.xlabel("t")
plt.ylabel("sum of density")
plt.title("Sum of density over time")
plt.savefig('density_over_time.png')
