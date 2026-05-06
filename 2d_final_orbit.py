import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

rng = np.random.default_rng()

#==========system=======
numberofatoms = 2
boxsize = 100
timestep = 0.001
equilibrium = 40
spring_k = 20
num_steps = int(input("How many steps to integrate? "))

# state vector: [x0, y0, x1, y1, vx0, vy0, vx1, vy1, ax0, ay0, ax1, ay1, 1]
# indices:
#   0,1  -> x0,  y0
#   2,3  -> x1,  y1
#   4,5  -> vx0, vy0
#   6,7  -> vx1, vy1
#   8,9  -> ax0, ay0
#  10,11 -> ax1, ay1
#  12    -> bias (always 1)

params = np.ones((13, 1))
init_vals = rng.random((8, 1)) * boxsize
params[:8] = init_vals

#custom values
x0_i,  y0_i  =  5.0, 50.0
x1_i,  y1_i  = 95.0, 50.0
vx0_i, vy0_i =  0.0,  40.0
vx1_i, vy1_i =  0.0, -40.0

params = np.zeros((13, 1))
params[ 0, 0] = x0_i;  params[ 1, 0] = y0_i
params[ 2, 0] = x1_i;  params[ 3, 0] = y1_i
params[ 4, 0] = vx0_i; params[ 5, 0] = vy0_i
params[ 6, 0] = vx1_i; params[ 7, 0] = vy1_i
params[12, 0] = 1.0

print("initial:")
print(params)

# spring update: writes ax0,ay0,ax1,ay1 from current positions.
# the 2d spring force F = k*(dist-eq)*r_hat is nonlinear because r_hat
# depends on dist. we factor it as:
#
#   F  = k * (dist - eq) * r_hat
#      = k * dist * r_hat  -  k * eq * r_hat
#      = k * (x1-x0, y1-y0)  -  k * eq * (rx, ry)
#
# the first term is linear in the position entries of the state vector.
# the second term is a constant bias for this step, encoded in the last
# column (via the augmented 1), scaled by the current r_hat.
# r_hat = (rx, ry) is recomputed each step before building this matrix.
#
#   ax0 =  k*(x1-x0)*rx/|r| ... simplified: k*(x1-x0) - k*eq*rx
#   ay0 =  k*(y1-y0)*ry/|r| ... simplified: k*(y1-y0) - k*eq*ry  <- wait
#
# more carefully:
#   ax0 =  k * (dist - eq) * rx  =  k*dist*rx - k*eq*rx
#         =  k*(x1-x0)           - k*eq*rx          (since dist*rx = x1-x0)
#   ay0 =  k*(y1-y0)             - k*eq*ry
#   ax1 = -k*(x1-x0)             + k*eq*rx
#   ay1 = -k*(y1-y0)             + k*eq*ry

def make_spring_update(rx, ry):
    """
    rx, ry: unit vector from particle 0 to particle 1 (recomputed each step).
    returns the 13x13 spring_update matrix for this step.
    """
    spring_update = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],                                         # x0  unchanged
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],                                         # y0  unchanged
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],                                         # x1  unchanged
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],                                         # y1  unchanged
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],                                         # vx0 unchanged
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],                                         # vy0 unchanged
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],                                         # vx1 unchanged
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],                                         # vy1 unchanged
        [-spring_k,         0, spring_k,        0, 0, 0, 0, 0, 0, 0, 0, 0, -spring_k*equilibrium*rx],  # ax0
        [0,        -spring_k,         0, spring_k, 0, 0, 0, 0, 0, 0, 0, 0, -spring_k*equilibrium*ry],  # ay0
        [ spring_k,         0, -spring_k,        0, 0, 0, 0, 0, 0, 0, 0, 0,  spring_k*equilibrium*rx],  # ax1
        [0,         spring_k,          0, -spring_k, 0, 0, 0, 0, 0, 0, 0, 0, spring_k*equilibrium*ry],  # ay1
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],                                         # bias row, always 1
    ])
    return spring_update

# velocity update: vx0 += ax0*dt, vy0 += ay0*dt, vx1 += ax1*dt, vy1 += ay1*dt
updatev = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0, 0,        0,        0,        0,        0],   # x0  unchanged
    [0, 1, 0, 0, 0, 0, 0, 0, 0,        0,        0,        0,        0],   # y0  unchanged
    [0, 0, 1, 0, 0, 0, 0, 0, 0,        0,        0,        0,        0],   # x1  unchanged
    [0, 0, 0, 1, 0, 0, 0, 0, 0,        0,        0,        0,        0],   # y1  unchanged
    [0, 0, 0, 0, 1, 0, 0, 0, timestep, 0,        0,        0,        0],   # vx0 += ax0*dt
    [0, 0, 0, 0, 0, 1, 0, 0, 0,        timestep, 0,        0,        0],   # vy0 += ay0*dt
    [0, 0, 0, 0, 0, 0, 1, 0, 0,        0,        timestep, 0,        0],   # vx1 += ax1*dt
    [0, 0, 0, 0, 0, 0, 0, 1, 0,        0,        0,        timestep, 0],   # vy1 += ay1*dt
    [0, 0, 0, 0, 0, 0, 0, 0, 1,        0,        0,        0,        0],   # ax0 unchanged
    [0, 0, 0, 0, 0, 0, 0, 0, 0,        1,        0,        0,        0],   # ay0 unchanged
    [0, 0, 0, 0, 0, 0, 0, 0, 0,        0,        1,        0,        0],   # ax1 unchanged
    [0, 0, 0, 0, 0, 0, 0, 0, 0,        0,        0,        1,        0],   # ay1 unchanged
    [0, 0, 0, 0, 0, 0, 0, 0, 0,        0,        0,        0,        1],   # bias row
])
print("velocity update:")
print(updatev)

# position update: x0 += vx0*dt, y0 += vy0*dt, x1 += vx1*dt, y1 += vy1*dt
updatex = np.array([
    [1, 0, 0, 0, timestep, 0,        0,        0,        0, 0, 0, 0, 0],   # x0 += vx0*dt
    [0, 1, 0, 0, 0,        timestep, 0,        0,        0, 0, 0, 0, 0],   # y0 += vy0*dt
    [0, 0, 1, 0, 0,        0,        timestep, 0,        0, 0, 0, 0, 0],   # x1 += vx1*dt
    [0, 0, 0, 1, 0,        0,        0,        timestep, 0, 0, 0, 0, 0],   # y1 += vy1*dt
    [0, 0, 0, 0, 1,        0,        0,        0,        0, 0, 0, 0, 0],   # vx0 unchanged
    [0, 0, 0, 0, 0,        1,        0,        0,        0, 0, 0, 0, 0],   # vy0 unchanged
    [0, 0, 0, 0, 0,        0,        1,        0,        0, 0, 0, 0, 0],   # vx1 unchanged
    [0, 0, 0, 0, 0,        0,        0,        1,        0, 0, 0, 0, 0],   # vy1 unchanged
    [0, 0, 0, 0, 0,        0,        0,        0,        1, 0, 0, 0, 0],   # ax0 unchanged
    [0, 0, 0, 0, 0,        0,        0,        0,        0, 1, 0, 0, 0],   # ay0 unchanged
    [0, 0, 0, 0, 0,        0,        0,        0,        0, 0, 1, 0, 0],   # ax1 unchanged
    [0, 0, 0, 0, 0,        0,        0,        0,        0, 0, 0, 1, 0],   # ay1 unchanged
    [0, 0, 0, 0, 0,        0,        0,        0,        0, 0, 0, 0, 1],   # bias row
])
print("position update:")
print(updatex)

# master update: apply velocity then position (spring_update is built fresh
# each step since r_hat is nonlinear, but master_update is constant)
master_update = updatex @ updatev
print("master update:")
print(master_update)

# history stores the 12 physical values (drop the bias row)
history = np.zeros((12, num_steps))

#==========integration loop=======
for step in range(num_steps):
    # recompute r_hat, build spring_update, write new accelerations into params
    dx = params[2, 0] - params[0, 0]
    dy = params[3, 0] - params[1, 0]
    dist = math.sqrt(dx*dx + dy*dy)
    if dist > 0:
        rx, ry = dx / dist, dy / dist
    else:
        rx, ry = 0.0, 0.0
    spring_update = make_spring_update(rx, ry)
    params = spring_update @ params

    # integrate: velocity then position in one matrix multiply
    params = master_update @ params

    # boundary conditions on x0
    if params[0, 0] <= 0:
        params[0, 0] *= -1
        params[4, 0] *= -1
        params[8, 0] *= -1
    elif params[0, 0] >= boxsize:
        params[0, 0] = 2*boxsize - params[0, 0]
        params[4, 0] *= -1
        params[8, 0] *= -1

    # boundary conditions on y0
    if params[1, 0] <= 0:
        params[1, 0] *= -1
        params[5, 0] *= -1
        params[9, 0] *= -1
    elif params[1, 0] >= boxsize:
        params[1, 0] = 2*boxsize - params[1, 0]
        params[5, 0] *= -1
        params[9, 0] *= -1

    # boundary conditions on x1
    if params[2, 0] <= 0:
        params[2, 0] *= -1
        params[6, 0] *= -1
        params[10, 0] *= -1
    elif params[2, 0] >= boxsize:
        params[2, 0] = 2*boxsize - params[2, 0]
        params[6, 0] *= -1
        params[10, 0] *= -1

    # boundary conditions on y1
    if params[3, 0] <= 0:
        params[3, 0] *= -1
        params[7, 0] *= -1
        params[11, 0] *= -1
    elif params[3, 0] >= boxsize:
        params[3, 0] = 2*boxsize - params[3, 0]
        params[7, 0] *= -1
        params[11, 0] *= -1

    history[:, step] = params[:12, 0]

print("Integration complete.")
print("History array shape:", history.shape)

#==========animation=======
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 6))

ax.set_xlim(0, boxsize)
ax.set_ylim(0, boxsize)
ax.set_aspect('equal')
ax.set_xlabel("X Position")
ax.set_ylabel("Y Position")
ax.set_title("2D Two-Particle Simulation (matrix-based spring force)")
for spine in ax.spines.values():
    spine.set_linewidth(2)

colors = ['blue', 'red']
particles = [ax.plot([], [], 'o', color=colors[p], markersize=12)[0] for p in range(numberofatoms)]
trails    = [ax.plot([], [], '-', color=colors[p], alpha=0.3, linewidth=1)[0] for p in range(numberofatoms)]
spring_line, = ax.plot([], [], 'g-', linewidth=1.5, alpha=0.6)

step_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, va='top')
info_texts = [ax.text(0.02, 0.91 - p*0.09, '', transform=ax.transAxes, color=colors[p], fontsize=8)
              for p in range(numberofatoms)]

ax2.set_xlim(0, num_steps)
ax2.set_ylim(0, boxsize * math.sqrt(2))
ax2.axhline(equilibrium, color='green', linestyle='--', linewidth=1, label=f'equilibrium ({equilibrium})')
ax2.set_xlabel("Step")
ax2.set_ylabel("Distance")
ax2.set_title("Inter-particle Distance")
ax2.legend(loc='upper right')
dist_line, = ax2.plot([], [], 'k-', linewidth=1)

trail_length = 20

def init():
    for p in range(numberofatoms):
        particles[p].set_data([], [])
        trails[p].set_data([], [])
        info_texts[p].set_text('')
    spring_line.set_data([], [])
    step_text.set_text('')
    dist_line.set_data([], [])
    return particles + trails + info_texts + [step_text, spring_line, dist_line]

def animate(i):
    step_text.set_text(f'Step: {i+1}/{num_steps}')
    x0, y0 = history[0, i], history[1, i]
    x1, y1 = history[2, i], history[3, i]
    vx0, vy0 = history[4, i], history[5, i]
    vx1, vy1 = history[6, i], history[7, i]
    ax0, ay0 = history[8, i], history[9, i]
    ax1, ay1 = history[10, i], history[11, i]

    particles[0].set_data([x0], [y0])
    particles[1].set_data([x1], [y1])

    start = max(0, i - trail_length)
    trails[0].set_data(history[0, start:i+1], history[1, start:i+1])
    trails[1].set_data(history[2, start:i+1], history[3, start:i+1])

    info_texts[0].set_text(f'P1  x:{x0:.1f}  y:{y0:.1f}  vx:{vx0:.2f}  vy:{vy0:.2f}  ax:{ax0:.2f}  ay:{ay0:.2f}')
    info_texts[1].set_text(f'P2  x:{x1:.1f}  y:{y1:.1f}  vx:{vx1:.2f}  vy:{vy1:.2f}  ax:{ax1:.2f}  ay:{ay1:.2f}')

    spring_line.set_data([x0, x1], [y0, y1])

    distances = np.sqrt((history[2, :i+1] - history[0, :i+1])**2 +
                        (history[3, :i+1] - history[1, :i+1])**2)
    dist_line.set_data(range(i+1), distances)

    return particles + trails + info_texts + [step_text, spring_line, dist_line]

ani = animation.FuncAnimation(
    fig, animate, frames=num_steps,
    init_func=init, interval=1, blit=True
)

plt.tight_layout()
plt.show()
