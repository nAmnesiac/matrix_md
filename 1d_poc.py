#======IMPORTING LIBRARIES===================================================#
import math
import numpy as np
import sys
import random
import os.path
#======END===================================================================#

#==========system=======
numberofatoms = 3
boxsize = 100
mass = 1


coords = np.random.uniform(0, numberofatoms, size=(1, 3))
