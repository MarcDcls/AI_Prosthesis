import numpy
from math import *

def interpolate(posHandStart, posHandEnd, step):
    return numpy.linspace(posHandStart, posHandEnd, step)

print(interpolate([0,0,0,0,pi],[5,10,20,pi,2*pi],5))
