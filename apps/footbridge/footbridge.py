import sys
sys.path.append('../../')
from digital_twin import *
from src.model.ccx.ccx_model import *
from src.diagnosis.dt_diagnosis import system_identification
import numpy as np
import os

##
##  footbridge.py
##

nsensor = 36
young   = 2.0E+11
poisson = 0.3
density = 7.8E+03

# Set up sensor locations
def set_sensors(model):
    xsensor = np.zeros(nsensor, dtype=np.float64)
    ysensor = np.zeros(nsensor, dtype=np.float64)
    zsensor = np.zeros(nsensor, dtype=np.float64)
    model.set_nsensor(nsensor)
    isensor = -1
    xs = -6.0
    zs = 0.0
    ys = 0.0

    while isensor < 23:
        xs += 8.0
        isensor += 1
        xsensor[isensor] = xs + 2.0
        ysensor[isensor] = 0.0
        zsensor[isensor] = 0.0
        isensor += 1
        xsensor[isensor] = xs + 2.0
        ysensor[isensor] = 4.0
        zsensor[isensor] = 0.0
        isensor += 1
        xsensor[isensor] = xs
        ysensor[isensor] = 0.0
        zsensor[isensor] = 3.5
        isensor += 1
        xsensor[isensor] = xs
        ysensor[isensor] = 4.0
        zsensor[isensor] = 3.5

    xs = -2.0
    while isensor < nsensor-1:
        xs += 4.0
        isensor += 1
        xsensor[isensor] = xs
        ysensor[isensor] = 2.0
        zsensor[isensor] = 0.0

    for isensor in range(nsensor):
        x = xsensor[isensor]
        y = ysensor[isensor]
        z = zsensor[isensor]
        model.add_sensor_location(isensor, x, y, z)

# Set up synthetic target
def set_target(model):
    strength_factor = np.ones(model.nelem, dtype=np.float64)
    # Add weakening
    strength_factor[10]  = 0.5
    strength_factor[11]  = 0.5
    strength_factor[12]  = 0.5
    strength_factor[13]  = 0.5
    strength_factor[14]  = 0.5
    strength_factor[15]  = 0.5
    strength_factor[16]  = 0.5
    strength_factor[17]  = 0.5
    strength_factor[18]  = 0.5
    strength_factor[19]  = 0.5
    strength_factor[20]  = 0.5
    strength_factor[21]  = 0.5
    strength_factor[188] = 0.5
    strength_factor[189] = 0.5
    strength_factor[190] = 0.5
    strength_factor[191] = 0.5
    strength_factor[338] = 0.5
    strength_factor[339] = 0.5
    strength_factor[340] = 0.5
    strength_factor[341] = 0.5
    strength_factor[432] = 0.5
    strength_factor[433] = 0.5
    strength_factor[434] = 0.5
    model.target(strength_factor)
    
if __name__ == "__main__":

    os.system('rm -r results')
    
    dt = digital_twin("footbridge")

    print(dt)

    dt.model = ccx_model({'nproc_ccx': 8})

    set_sensors(dt.model)
    
    set_target(dt.model)

    system_identification(dt.model, pyrol_file="pyrol_input.xml")
