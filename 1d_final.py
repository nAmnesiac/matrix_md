import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import os.path

rng = np.random.default_rng()

#==========system=======
numberofatoms = 2
boxsize = 100
timestep = 0.001
equilibrium = 40
spring_k = 15
num_steps = int(input("How many steps to integrate? "))

# state vector: [x0, x1, v0, v1, a0, a1, 1]
# the final row is always 1, used to encode the equilibrium offset
params = np.ones((7, 1))
init_vals = rng.random((6, 1)) * boxsize
params[:6] = init_vals
print("initial:")
print(params)

# spring matrix: recomputes a0 and a1 from current positions
# a0 =  k*(x1-x0) - k*eq
# a1 = -k*(x1-x0) + k*eq
# the last column encodes the constant equilibrium bias via the augmented 1
spring_update = np.array([
    [1, 0, 0, 0, 0, 0, 0],                          # x0 unchanged
    [0, 1, 0, 0, 0, 0, 0],                          # x1 unchanged
    [0, 0, 1, 0, 0, 0, 0],                          # v0 unchanged
    [0, 0, 0, 1, 0, 0, 0],                          # v1 unchanged
    [-spring_k, spring_k, 0, 0, 0, 0, -spring_k*equilibrium],  # a0
    [ spring_k,-spring_k, 0, 0, 0, 0,  spring_k*equilibrium],  # a1
    [0, 0, 0, 0, 0, 0, 1],                          # bias row, always 1
])
print("spring update:")
print(spring_update)

# velocity update: v0 += a0*dt, v1 += a1*dt
updatev = np.array([
    [1, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, timestep, 0, 0],
    [0, 0, 0, 1, 0, timestep, 0],
    [0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 1],
])
print("velocity update:")
print(updatev)

# position update: x0 += v0*dt, x1 += v1*dt
updatex = np.array([
    [1, 0, timestep, 0, 0, 0, 0],
    [0, 1, 0, timestep, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 1],
])
print("position update:")
print(updatex)

# master update: apply spring, then velocity, then position
master_update = updatex @ updatev @ spring_update
print("master update:")
print(master_update)

# history stores the 6 physical values (drop the bias row)
history = np.zeros((6, num_steps))

#==========integration loop=======
for step in range(num_steps):
    params = master_update @ params

    # boundary conditions on x0
    if params[0, 0] <= 0:
        params[0, 0] *= -1
        params[2, 0] *= -1
        params[4, 0] *= -1
    elif params[0, 0] >= boxsize:
        params[0, 0] = 2*boxsize - params[0, 0]
        params[2, 0] *= -1
        params[4, 0] *= -1

    # boundary conditions on x1
    if params[1, 0] <= 0:
        params[1, 0] *= -1
        params[3, 0] *= -1
        params[5, 0] *= -1
    elif params[1, 0] >= boxsize:
        params[1, 0] = 2*boxsize - params[1, 0]
        params[3, 0] *= -1
        params[5, 0] *= -1

    history[:, step] = params[:6, 0]

print("Integration complete.")
print("History array shape:", history.shape)

#==========animation=======
fig, (ax, ax2) = plt.subplots(2, 1, figsize=(8, 6),
                               gridspec_kw={'height_ratios': [3, 1]})

ax.set_xlim(0, boxsize)
ax.set_ylim(-0.5, 0.5)
ax.set_xlabel("Position")
ax.set_title("1D Two-Particle Simulation (fully matrix-based spring force)")
ax.axvline(0, color='black', linewidth=2)
ax.axvline(boxsize, color='black', linewidth=2)

colors = ['blue', 'red']
particles = [ax.plot([], [], 'o', color=colors[p], markersize=12)[0] for p in range(numberofatoms)]
trails    = [ax.plot([], [], '-', color=colors[p], alpha=0.3, linewidth=1)[0] for p in range(numberofatoms)]
spring_line, = ax.plot([], [], 'g-', linewidth=1.5, alpha=0.6)

step_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)
info_texts = [ax.text(0.02, 0.75 - p*0.20, '', transform=ax.transAxes, color=colors[p])
              for p in range(numberofatoms)]

ax2.set_xlim(0, num_steps)
ax2.set_ylim(0, boxsize)
ax2.axhline(equilibrium, color='green', linestyle='--', linewidth=1, label=f'equilibrium ({equilibrium})')
ax2.set_xlabel("Step")
ax2.set_ylabel("Distance")
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
    x0, x1 = history[0, i], history[1, i]
    v0, v1 = history[2, i], history[3, i]
    a0, a1 = history[4, i], history[5, i]

    particles[0].set_data([x0], [0])
    particles[1].set_data([x1], [0])

    for p, (x, v, a) in enumerate([(x0,v0,a0),(x1,v1,a1)]):
        start = max(0, i - trail_length)
        trails[p].set_data(history[p, start:i+1], np.zeros(i+1 - start))
        info_texts[p].set_text(f'P{p+1}  x:{x:.1f}  v:{v:.2f}  a:{a:.2f}')

    spring_line.set_data([x0, x1], [0, 0])

    distances = abs(history[1, :i+1] - history[0, :i+1])
    dist_line.set_data(range(i+1), distances)

    return particles + trails + info_texts + [step_text, spring_line, dist_line]

ani = animation.FuncAnimation(
    fig, animate, frames=num_steps,
    init_func=init, interval=2, blit=True
)

plt.tight_layout()
plt.show()
