import sys
sys.path.append('../../')
from digital_twin import *
from src.model.ccx.ccx_model import *
from src.diagnosis.dt_diagnosis import system_identification
import numpy as np
import os

##
##  simple_truss.py
##

# Plate Geometry and Properties
h  = 1.0
lx = 10.0
nx = int(lx/h)
young   = 2.0e+11
poisson = 0.3
density = 7.8e+03

# Set up sensor locations
def set_sensors(model):
    nx = 10
    model.set_nsensor(nx)
    model.add_sensor_location(0, 1.0, 0.0, 0.0)
    model.add_sensor_location(1, 2.0, 0.0, 0.0)
    model.add_sensor_location(2, 3.0, 0.0, 0.0)
    model.add_sensor_location(3, 4.0, 0.0, 0.0)
    model.add_sensor_location(4, 5.0, 0.0, 0.0)
    model.add_sensor_location(5, 6.0, 0.0, 0.0)
    model.add_sensor_location(6, 7.0, 0.0, 0.0)
    model.add_sensor_location(7, 8.0, 0.0, 0.0)
    model.add_sensor_location(8, 9.0, 0.0, 0.0)
    model.add_sensor_location(9, 10.0, 0.0, 0.0)

# Set up synthetic target
def set_target(model):
    strength_factor = np.ones(model.nelem, dtype=np.float64)
    strength_factor[4] = 0.25
    model.target(strength_factor)
    
if __name__ == "__main__":

    os.system('rm -r results')

    dt = digital_twin("truss")

    print(dt)
    input_vars = {'nsmoothing': 0,
                  'nproc_ccx': 1,
                  'nproc_cases': 1,
                  'run_type': 'dynamic',
                  'ntime': 101,
                  'dt': 1.0E-02}
    dt.model = ccx_model(input_vars)

    set_sensors(dt.model)

    set_target(dt.model)

    system_identification(dt, pyrol_file="pyrol_input.xml")
