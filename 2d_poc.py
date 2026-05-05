import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import os.path

rng = np.random.default_rng()

#==========system=======
numberofatoms = 1
boxsize = 100
timestep = 0.5
num_steps = int(input("How many steps to integrate? "))

# params is now a 6x1 vector: [x, y, vx, vy, ax, ay]
params = rng.random((6,1))
params *= boxsize
params[2,0] = 1  # vx
params[3,0] = 1  # vy
params[4,0] = 1  # ax
params[5,0] = 1  # ay

print("initial:")
print(params)

# velocity update matrix (6x6)
# vx += ax*dt, vy += ay*dt, everything else unchanged
updatev = np.array([
    [1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, timestep, 0],
    [0, 0, 0, 1, 0, timestep],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1]
])

# position update matrix (6x6)
# x += vx*dt, y += vy*dt, everything else unchanged
updatex = np.array([
    [1, 0, timestep, 0, 0, 0],
    [0, 1, 0, timestep, 0, 0],
    [0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1]
])

master_update = updatex @ updatev
print("master update")
print(master_update)

# history matrix: 6 rows x num_steps columns
history = np.zeros((6, num_steps))

#==========integration loop=======
for step in range(num_steps):
    params = master_update @ params

    # x boundary
    if params[0,0] <= 0:
        params[0,0] *= -1
        params[2,0] *= -1
        params[4,0] *= -1
    elif params[0,0] >= boxsize:
        params[0,0] = 2*boxsize - params[0,0]
        params[2,0] *= -1
        params[4,0] *= -1

    # y boundary
    if params[1,0] <= 0:
        params[1,0] *= -1
        params[3,0] *= -1
        params[5,0] *= -1
    elif params[1,0] >= boxsize:
        params[1,0] = 2*boxsize - params[1,0]
        params[3,0] *= -1
        params[5,0] *= -1

    history[:, step] = params[:, 0]

print("Integration complete.")
print("History matrix shape:", history.shape)

#==========animation=======
fig, ax = plt.subplots()
ax.set_xlim(0, boxsize)
ax.set_ylim(0, boxsize)
ax.set_xlabel("X Position")
ax.set_ylabel("Y Position")
ax.set_title("2D Particle Simulation")
ax.set_aspect('equal')

particle, = ax.plot([], [], 'bo', markersize=12)
trail, = ax.plot([], [], 'b-', alpha=0.3, linewidth=1)
step_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
vel_text  = ax.text(0.02, 0.88, '', transform=ax.transAxes)
acc_text  = ax.text(0.02, 0.81, '', transform=ax.transAxes)

trail_length = 20

def init():
    particle.set_data([], [])
    trail.set_data([], [])
    step_text.set_text('')
    vel_text.set_text('')
    acc_text.set_text('')
    return particle, trail, step_text, vel_text, acc_text

def animate(i):
    x  = history[0, i]
    y  = history[1, i]
    vx = history[2, i]
    vy = history[3, i]
    ax_ = history[4, i]
    ay  = history[5, i]

    particle.set_data([x], [y])

    # trailing path
    start = max(0, i - trail_length)
    trail.set_data(history[0, start:i+1], history[1, start:i+1])

    step_text.set_text(f'Step: {i+1}/{num_steps}')
    vel_text.set_text(f'Velocity: ({vx:.2f}, {vy:.2f})')
    acc_text.set_text(f'Accel:    ({ax_:.2f}, {ay:.2f})')
    return particle, trail, step_text, vel_text, acc_text

ani = animation.FuncAnimation(
    fig, animate, frames=num_steps,
    init_func=init, interval=100, blit=True
)

plt.tight_layout()
plt.show()
