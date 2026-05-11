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
numberofatoms = 2
boxsize = 100
mass = 1

#each step, the disctance between the two particles is halved
#and they move closer to each other

coords = rng.random((numberofatoms,1)) #random generated within boxsize
coords *= boxsize

#coords = np.array([0, 100]) #for test purposes

stepper = np.array([[0.75, 0.25], 
                   [0.25, 0.75]]) #halves distance between particles 


print("initial")
print(coords)

#stupid test loop
while True:
    ask = 0 
    ask = input("1 to continue: ")
    if int(ask) == 1:
        coords = stepper @ coords
        print(coords)
    elif int(ask) == 67:
        break


print("done")



