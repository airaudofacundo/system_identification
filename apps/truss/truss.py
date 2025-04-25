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
    strength_factor[4] = 0.6
    model.target(strength_factor)

if __name__ == "__main__":

    os.system('rm -r results')

    dt = digital_twin("truss")

    print(dt)

    input_vars = {'ncase': 1,
                  'nproc_ccx': 1,
                  'nproc_cases': 1,
                  'nsmoothing': 0,
                  'run_type': 'static',
                  'obj_type': 'basic',
                  'ngauss': 0,
                  'nsample': 0,
                  'nrandom': 0,
                  'cvar_beta': 0.0,
                  'risk_lambda': 0.0,
                  'dt': 0.0,
                  'ntime': 1,
                  'gradient_factor': 1.0}
    dt.model = ccx_model(input_vars)

    set_sensors(dt.model)

    set_target(dt.model)

    system_identification(dt.model, pyrol_file="pyrol_input.xml")
