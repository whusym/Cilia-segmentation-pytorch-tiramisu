import numpy as np
from sys import *
import scipy.misc


mask=scipy.misc.imread(argv[1]+".png")
mask[mask==1] = 0

with open(argv[1]+".txt", "wb") as f:
    np.savetxt(f, mask.astype(int), fmt='%i', delimiter=",")
