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
num_steps = int(input("How many steps to integrate? "))

# params is now 3x2: rows are [x, v, a], columns are [particle1, particle2]
params = rng.random((3, numberofatoms))
params *= boxsize
#params[2,0] = 100 #for testing purposes
print("initial:")
print(params)

# update matrices unchanged
updatev = np.array([[1,0,0],[0,1,timestep],[0,0,1]])
updatex = np.array([[1,timestep,0],[0,1,0],[0,0,1]])
master_update = updatex @ updatev
print("master update")
print(master_update)

history = np.zeros((3, numberofatoms, num_steps))

#==========integration loop=======
for step in range(num_steps):
    params = master_update @ params

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
fig, ax = plt.subplots()
ax.set_xlim(0, boxsize)
ax.set_ylim(-0.5, 0.5)
ax.set_xlabel("Position")
ax.set_title("1D Two-Particle Simulation (constant force)")
ax.axvline(0, color='black', linewidth=2)
ax.axvline(boxsize, color='black', linewidth=2)

colors = ['blue', 'red']
particles = [ax.plot([], [], 'o', color=colors[p], markersize=12)[0] for p in range(numberofatoms)]
trails    = [ax.plot([], [], '-', color=colors[p], alpha=0.3, linewidth=1)[0] for p in range(numberofatoms)]

step_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)
info_texts = [ax.text(0.02, 0.75 - p*0.20, '', transform=ax.transAxes, color=colors[p])
              for p in range(numberofatoms)]

trail_length = 20

def init():
    for p in range(numberofatoms):
        particles[p].set_data([], [])
        trails[p].set_data([], [])
        info_texts[p].set_text('')
    step_text.set_text('')
    return particles + trails + info_texts + [step_text]

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

    return particles + trails + info_texts + [step_text]

ani = animation.FuncAnimation(
    fig, animate, frames=num_steps,
    init_func=init, interval=20, blit=True
)

plt.tight_layout()
plt.show()
