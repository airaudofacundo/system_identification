import sys
sys.path.append('../../')
from digital_twin import *
from src.model.ccx.ccx_model import *
from src.diagnosis.dt_diagnosis import system_identification
import numpy as np
import os

##
##  beam_3d.py
##

nsensor = 20

lx = 1.0
ly = 0.1
lz = 0.1

# Set up sensor locations
def set_sensors(model):
    model.set_nsensor(nsensor)

    isensor = -1
    x = -0.1
    for iseg in range(5):
        x += 0.2
        y = -0.05
        z = 0.025
        isensor += 1
        model.add_sensor_location(isensor, x, y, z)
        y = 0.15
        z = 0.025
        isensor += 1
        model.add_sensor_location(isensor, x, y, z)

    x = 0.0
    for iseg in range(5):
        x += 0.2
        y = 0.05
        z = -0.05
        isensor += 1
        model.add_sensor_location(isensor, x, y, z)
        y = 0.05
        z = 0.15
        isensor += 1
        model.add_sensor_location(isensor, x, y, z)

# Set up synthetic target
def set_target(model):
    strength_factor = np.ones(model.nelem, dtype=np.float64)
    # Add weakening
    for ielem in range(12252,12577):
        strength_factor[ielem] = 0.3
    model.target(strength_factor)
    
if __name__ == "__main__":

    os.system('rm -r results')
    
    dt = digital_twin("Beam 3D")

    print(dt)

    dt.model = ccx_model({'ncase': 2,
                          'nsmoothing': 5,
                          'nproc_ccx': 8,
                          'nproc_cases': 2,
                          'run_type': 'dynamic',
                          'ntime': 101,
                          'dt': 5.0E-03})

    set_sensors(dt.model)
    
    set_target(dt.model)

    system_identification(dt.model, pyrol_file="pyrol_input.xml")
