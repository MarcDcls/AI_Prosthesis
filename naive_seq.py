import numpy
from math import *
from data import get_current_target_naive_seq

def interpolate(posHandStart, posHandEnd, step):
    return numpy.linspace(posHandStart, posHandEnd, step)

print(interpolate([0,0,0,0,pi],[5,10,20,pi,2*pi],5))


def generate_seq():
    current, target, counts = get_current_target_naive_seq()
    result = map(interpolate, current[:,3:], target[:,3:], counts)
    
    