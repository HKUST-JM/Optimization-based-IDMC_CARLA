import carla
import math
import time

import numpy as np
from scipy.optimize import minimize

import casadi as ca

kf = -128916  # N/rad
kr = -85944  # N/rad
lf = 1.06  # m
lr = 1.85  # m
m = 1412  # kg
Iz = 1536.7  # kg*m2s
dt = 0.1
Lk = lf*kf - lr*kr  # = 22345.439999999973

n_input = 2  # acc, steering

## to be released