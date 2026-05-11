#this is an initial one dimensional proof of concept


#======IMPORTING LIBRARIES===================================================#
import math
import numpy as np
import sys
import random
import os.path
rng = np.random.default_rng()
#======END===================================================================#

#==========system=======
numberofatoms = 1
boxsize = 100
timestep = 0.5 #seconds

# a=a, vf=vi+at, xf=x0+at

params = rng.random((3,1)) #random generated within boxsize
params *= boxsize

params[1,0] = 1 #set velocity to 1 unit/second/second
params[2,0] = 1 #set acceleration to 1 unit/second/second

print("initial:")
print(params)

updatev = np.array([[1,0,0],[0,1,timestep],[0,0,1]])
print("v update")
print(updatev)

updatex = np.array([[1,timestep,0],[0,1,0],[0,0,1]])
print("x update")
print(updatex)

print("initial")
print(params)

#stupid test loop
while True:
    ask = 0 
    ask = input("1 to continue: ")
    if int(ask) == 1:
        print("updating v")
        params = updatev @ params
        print(params)

        print("updating x")
        params = updatex @ params
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


print("done")



