import sys
sys.path.append('../../')
import os
from digital_twin import *
from src.model.ccx.ccx_model import *
from src.diagnosis.dt_diagnosis import system_identification
import numpy as np

##
##  crane.py
##
##  3D Crane with beam elements.
##

# Parameters
nsensor = 10

# Set up sensor locations

def setup_sensors(model):
    model.set_nsensor(nsensor)
    if nsensor == 10:
        # 1, 14, 42, 48, 87, 103, 115, 35, 59, 28
        model.set_sensor(0, 0)
        model.set_sensor(1, 13)
        model.set_sensor(2, 27)
        model.set_sensor(3, 34)
        model.set_sensor(4, 41)
        model.set_sensor(5, 47)
        model.set_sensor(6, 58)
        model.set_sensor(7, 86)
        model.set_sensor(8, 102)
        model.set_sensor(9, 114)
    elif nsensor == 20:
        # 1, 14, 42, 48, 87, 103, 115, 35, 59, 28
        model.set_sensor(0, 0)
        model.set_sensor(1, 13)
        model.set_sensor(2, 27)
        model.set_sensor(3, 34)
        model.set_sensor(4, 41)
        model.set_sensor(5, 47)
        model.set_sensor(6, 58)
        model.set_sensor(7, 86)
        model.set_sensor(8, 102)
        model.set_sensor(9, 114)
        model.set_sensor(10, 1)
        model.set_sensor(11, 14)
        model.set_sensor(12, 28)
        model.set_sensor(13, 35)
        model.set_sensor(14, 42)
        model.set_sensor(15, 48)
        model.set_sensor(16, 59)
        model.set_sensor(17, 87)
        model.set_sensor(18, 103)
        model.set_sensor(19, 115)
    elif nsensor == model.npoin:
        for ipoin in range(model.npoin):
            model.set_sensor(ipoin, ipoin)
    else:
        print('WRONG nsensor (', nsensor, ')')

# Set up synthetic target
def set_target(model):
    strength_factor = np.ones(model.nelem, dtype=np.float64)
    # Column link
    strength_factor[50]  = 0.5
    strength_factor[51]  = 0.5
    strength_factor[141] = 0.5
    strength_factor[149] = 0.5
    strength_factor[175] = 0.5
    strength_factor[232] = 0.5
    strength_factor[259] = 0.5
    strength_factor[304] = 0.5
    model.target(strength_factor)
    
if __name__ == "__main__":

    os.system("rm -r results")
    
    dt = digital_twin("crane")

    print(dt)

    input_vars = {"ncase": 6,
                  "nproc_cases": 6}
    dt.model = ccx_model(input_vars)

    setup_sensors(dt.model)

    set_target(dt.model)

    system_identification(dt.model, pyrol_file="pyrol_input.xml")
