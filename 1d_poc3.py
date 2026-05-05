import math
import numpy as np
import sys
import random
import os.path
rng = np.random.default_rng()

#==========system=======
numberofatoms = 1
boxsize = 100
timestep = 0.5
params = rng.random((3,1))
params *= boxsize
params[1,0] = 1
params[2,0] = 1
print("initial:")
print(params)

updatev = np.array([[1,0,0],[0,1,timestep],[0,0,1]])
print("v update")
print(updatev)

updatex = np.array([[1,timestep,0],[0,1,0],[0,0,1]])
print("x update")
print(updatex)

master_update = updatex @ updatev
print("master update")
print(master_update)

print("initial")
print(params)

while True:
    ask = 0 
    ask = input("1 to continue: ")
    if int(ask) == 1:
        print("updating params")
        params = master_update @ params
        print(params)
        if params[0,0] <= 0:
            params = -1 * params
            print("negative outside box")
            print(params)
        elif params[0,0] >= boxsize:
            params[0,0] = 2*boxsize - params[0,0]
            params[1,0] *= -1
            params[2,0] *= -1
            print("positive outside box")
            print(params)
    elif int(ask) == 67:
        break

