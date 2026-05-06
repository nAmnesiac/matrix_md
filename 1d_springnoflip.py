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
timestep = 0.01
equilibrium = 40
spring_k = 10
num_steps = int(input("How many steps to integrate? "))

params = rng.random((3, numberofatoms))
params *= boxsize
print("initial:")
print(params)

updatev = np.array([[1,0,0],[0,1,timestep],[0,0,1]])
updatex = np.array([[1,timestep,0],[0,1,0],[0,0,1]])
master_update = updatex @ updatev
print("master update")
print(master_update)

history = np.zeros((3, numberofatoms, num_steps))

#==========integration loop=======
for step in range(num_steps):
    params = master_update @ params

    # spring force between particles
    displacement = params[0, 1] - params[0, 0]
    distance = abs(displacement)
    direction = np.sign(displacement)

    stretch = distance - equilibrium
    force = spring_k * stretch

    params[2, 0] =  direction * force
    params[2, 1] = -direction * force

    # boundary conditions — acceleration unchanged on wall collision
    for p in range(numberofatoms):
        if params[0, p] <= 0:
            params[0, p] *= -1
            params[1, p] *= -1
            # acceleration unchanged
        elif params[0, p] >= boxsize:
            params[0, p] = 2*boxsize - params[0, p]
            params[1, p] *= -1
            # acceleration unchanged

    history[:, :, step] = params

print("Integration complete.")
print("History array shape:", history.shape)

#==========animation=======
fig, (ax, ax2) = plt.subplots(2, 1, figsize=(8, 6),
                               gridspec_kw={'height_ratios': [3, 1]})

ax.set_xlim(0, boxsize)
ax.set_ylim(-0.5, 0.5)
ax.set_xlabel("Position")
ax.set_title("1D Two-Particle Simulation (spring force, constant acceleration)")
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
    for p in range(numberofatoms):
        x = history[0, p, i]
        v = history[1, p, i]
        a = history[2, p, i]

        particles[p].set_data([x], [0])

        start = max(0, i - trail_length)
        trails[p].set_data(history[0, p, start:i+1], np.zeros(i+1 - start))
        info_texts[p].set_text(f'P{p+1}  x:{x:.1f}  v:{v:.2f}  a:{a:.2f}')

    x0 = history[0, 0, i]
    x1 = history[0, 1, i]
    spring_line.set_data([x0, x1], [0, 0])

    distances = abs(history[0, 1, :i+1] - history[0, 0, :i+1])
    dist_line.set_data(range(i+1), distances)

    return particles + trails + info_texts + [step_text, spring_line, dist_line]

ani = animation.FuncAnimation(
    fig, animate, frames=num_steps,
    init_func=init, interval=20, blit=True
)

plt.tight_layout()
plt.show()
