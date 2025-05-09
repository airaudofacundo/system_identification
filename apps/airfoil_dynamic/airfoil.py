import sys
sys.path.append('../../')
from digital_twin import *
from src.model.ccx.ccx_model import *
from src.diagnosis.dt_diagnosis import system_identification
import numpy as np
import os

##
##  airfoil.py
##

nsensor = 40

# Set up sensor locations
def set_sensors(model):
    xsensor = np.zeros(nsensor, dtype=np.float64)
    ysensor = np.zeros(nsensor, dtype=np.float64)
    zsensor = np.zeros(nsensor, dtype=np.float64)
    model.set_nsensor(nsensor)
    isensor = -1
   
    if nsensor == 36:
        zcoords = [0.25, 0.75, 1.25, 1.75, 2.25, 2.75]
        xcoords = [0.25, 0.5, 0.75]
        ycoords = [0.058, 0.053, 0.03]

    if nsensor == 40:
        zcoords = [0.3, 1.1, 1.9, 2.7]
        xcoords = [0.1, 0.3, 0.5, 0.7, 0.9]
        ycoords = [0.047, 0.06, 0.0524, 0.036, 0.013]
        
    if nsensor == 60:
        zcoords = [0.25, 0.75, 1.25, 1.75, 2.25, 2.75]
        xcoords = [0.1, 0.3, 0.5, 0.7, 0.9]
        ycoords = [0.047, 0.06, 0.0524, 0.036, 0.013]

    for zc in zcoords:
        for idx, xc in enumerate(xcoords):
            isensor += 1
            xsensor[isensor] = xc
            ysensor[isensor] = ycoords[idx]
            zsensor[isensor] = zc

            isensor += 1
            xsensor[isensor] = xc
            ysensor[isensor] = -1*ycoords[idx]
            zsensor[isensor] = zc

    for isensor in range(nsensor):
        x = xsensor[isensor]
        y = ysensor[isensor]
        z = zsensor[isensor]
        model.add_sensor_location(isensor, x, y, z)


# Set up synthetic target
def set_target(model):
    strength_factor = np.ones(model.nelem, dtype=np.float64)
    # Add weakening
    for ielem in range(model.nelem):
        xc = 0.0
        yc = 0.0
        zc = 0.0
        for ipoin in model.element[ielem].point:
            xc += model.point_coord[ipoin][0]
            yc += model.point_coord[ipoin][1]
            zc += model.point_coord[ipoin][2]
        xc /= model.element[ielem].npoin
        yc /= model.element[ielem].npoin
        zc /= model.element[ielem].npoin
        if -0.01 < zc and zc <= 1.00:
            if 0.275 < xc and xc < 0.285:
                if abs(yc) < 0.059:
                    strength_factor[ielem] = 0.1
            elif 0.478 < xc and xc < 0.49:
                if abs(yc) < 0.054:
                    strength_factor[ielem] = 0.1
            # elif 0.10 < xc and xc < 0.11:
            #     if abs(yc) < 0.06:
            #         strength_factor[ielem] = 0.1
            # elif 0.73 < xc and xc < 0.74:
            #     if abs(yc) < 0.03:
            #         strength_factor[ielem] = 0.1
    model.target(strength_factor)

    
if __name__ == "__main__":

    os.system('rm -r results')
    
    dt = digital_twin("airfoil")

    print(dt)

    dt.model = ccx_model({'ncase': 1,
                          'nproc_ccx': 16,
                          'gradient_factor': 1.0e+00,
                          'do_vtx_morphing' : False,
                          'vm_radius'       : 0.0,
                          'nsmoothing': 5,
                          'run_type': 'dynamic',
                          'ntime': 51,
                          'dt': 0.01 })

    set_sensors(dt.model)
    
    set_target(dt.model)

    system_identification(dt.model,
                          input_vars={'lower_limit': 0.01,
                                      'upper_limit': 1.0},
                          pyrol_file="pyrol_input.xml")
