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
timestep = 5
num_steps = int(input("How many steps to integrate? "))

params = rng.random((3,1))
params *= boxsize
params[1,0] = 1
params[2,0] = 50
print("initial:")
print(params)

updatev = np.array([[1,0,0],[0,1,timestep],[0,0,1]])
updatex = np.array([[1,timestep,0],[0,1,0],[0,0,1]])
master_update = updatex @ updatev
print("master update")
print(master_update)

# history matrix: 3 rows (x, v, a) x num_steps columns
history = np.zeros((3, num_steps))

#==========integration loop=======
for step in range(num_steps):
    params = master_update @ params

    # boundary conditions
    if params[0,0] <= 0:
        params = -1 * params
    elif params[0,0] >= boxsize:
        params[0,0] = 2*boxsize - params[0,0]
        params[1,0] *= -1
        params[2,0] *= -1

    history[:, step] = params[:, 0]

print("Integration complete.")
print("History matrix shape:", history.shape)

#==========animation=======
fig, ax = plt.subplots()
ax.set_xlim(0, boxsize)
ax.set_ylim(-0.5, 0.5)
ax.set_xlabel("Position")
ax.set_title("1D Particle Simulation")
ax.axvline(0, color='black', linewidth=2)
ax.axvline(boxsize, color='black', linewidth=2)

particle, = ax.plot([], [], 'bo', markersize=12)
step_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)
vel_text  = ax.text(0.02, 0.75, '', transform=ax.transAxes)
acc_text  = ax.text(0.02, 0.60, '', transform=ax.transAxes)

def init():
    particle.set_data([], [])
    step_text.set_text('')
    vel_text.set_text('')
    acc_text.set_text('')
    return particle, step_text, vel_text, acc_text

def animate(i):
    x = history[0, i]
    v = history[1, i]
    a = history[2, i]
    particle.set_data([x], [0])
    step_text.set_text(f'Step: {i+1}/{num_steps}')
    vel_text.set_text(f'Velocity: {v:.3f}')
    acc_text.set_text(f'Accel: {a:.3f}')
    return particle, step_text, vel_text, acc_text

ani = animation.FuncAnimation(
    fig, animate, frames=num_steps,
    init_func=init, interval=100, blit=True
)

plt.tight_layout()
