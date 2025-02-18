import numpy as np
import os
import copy
import re
from src.model.dt_model import *
from src.model.ccx.calculix_element import *
from src.model.ccx.integrator import *
from src.model.ccx.sparsegrid import *
from src.model.ccx.ccx2paraview import generate_vtk
from scipy import stats
from scipy.stats import norm, skewnorm
from scipy.spatial import KDTree
from itertools import product
from src.model.ccx.sparsegrid import SparseInterpolator
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

"""
    ccx_model

        Class to use set up and run Calculix solver for system identification.

        Manual: https://www.dhondt.de/ccx_2.21.pdf
"""

class ccx_model(dt_model):

    # Initialize some class variables

    # Number of cases run simultaneously
    nproc_cases = 1

    # Number of processors for Calculix run
    nproc_ccx = 1

    # Smoothing passes
    nsmoothing = 0

    # FEM point coordinates
    point_coord = np.array([])

    # FEM Element (See calculix_element class)
    element = np.array([])

    # Boundary type and degrees of freedom
    boundary_type = {}
    boundary_dofs = {}

    # Sensor point list
    sensor_point = np.array([])

    # Strength factor we optimize for
    strength_factor = np.array([])

    # Keep track of sensor unknowns
    sensor_unkno_targ = np.array([])
    sensor_unkno_curr = np.array([])

    # Factor to multiply gradient and sensor loads by
    gradient_factor = 1.0

    # Objective type
    # obj_type == -1 == 'basic'
    #          ==  0 == 'neutral'
    #          ==  1 == 'cvar'
    #          ==  2 == 'neutral+cvar'
    #          ==  3 == 'mean+semideviation'
    #          ==  4 == 'entropic_risk'
    obj_type = -1

    # Determine whether there is uncertainty in the run
    is_uncertain = False

    # Determine whether run is static or synamic
    run_type = 'static'

    # Uncertainty variables

    # Number of samples needed to be computed each iteration
    nsample = 1

    # Beta value for CVaR case
    cvar_beta = 0.5

    # Lower limit for var. For use in CVaR case
    var_min = 1.0e-16

    # Initial value for var. For use in CVaR case
    var = 1.0e-06

    # Sparse grid intepolation parameters
    interpolation_nval = 2
    interpolated_obj = []

    # For sparse grid runs, need to track icase
    icase = 0

    """
    [1]  __init__

        General initializer for ccx_model.
        Calculix input files should be located at ccx_input/calculix_forward_case0.inp
                                                  ccx_input/calculix_forward_case1.inp
..
    """
    def __init__(self, input_vars={}):

        self.iter       = -1
        ncase           = input_vars.get('ncase', 1)
        nproc_ccx       = input_vars.get('nproc_ccx', 1)
        nproc_cases     = input_vars.get('nproc_cases', 1)
        nsmoothing      = input_vars.get('nsmoothing', 5)
        run_type        = input_vars.get('run_type', 'static')
        obj_type        = input_vars.get('obj_type', 'basic')
        ngauss          = input_vars.get('ngauss', 3)
        nsample         = input_vars.get('nsample', 1)
        nrandom         = input_vars.get('nrandom', 0)
        cvar_beta       = input_vars.get('cvar_beta', 0.1)
        risk_lambda     = input_vars.get('risk_lambda', 0.5)
        error_lower     = input_vars.get('error_lower', 0.8)
        error_upper     = input_vars.get('error_upper', 1.2)
        dt              = input_vars.get('dt', 0.0)
        ntime           = input_vars.get('ntime', 1)
        gradient_factor = input_vars.get('gradient_factor', 1.0)

        self.set_ncase(ncase)
        self.set_nproc(nproc_ccx, nproc_cases)
        self.set_objective_type(obj_type)

        self.run_type = run_type
        self.dt = dt
        self.ntime = ntime

        # To Do: Put this into read_input_file
        if ngauss > 0 and nrandom > 0:
            self.set_integration_order(ngauss, nrandom)
            self.set_cvar_beta(cvar_beta)
            self.set_risk_lambda(risk_lambda)

        # Only read mesh from first case.. Hope it's the same for all!
        self.read_inp_file("ccx_input/calculix_forward_case0.inp")

        self.compute_stiffness()

        self.strength_factor = np.ones(self.nelem, dtype=np.float64)

        self.nsmoothing = nsmoothing


    """
    [2]  read_inp_file

        Initializes variables from a calculix input file '.inp'.
        Currently only admits one load case
    """
    def read_inp_file(self, ccx_file):
        with open(ccx_file, 'r') as file:
            material_elastic = {}
            material_poisson = {}
            material_density = {}
            element_type = {}
            elset_list = {}
            nset_list = {}
            elmat = {}
            connections_list = {}
            element_types = {}
            coordinates_list = {}
            section_types = {}
            beam_rect = {}
            beam_box = {}
            shell_thickness = {}


            for line in file:
                line = line.strip()

                # Check for keywords
                if line.startswith('*'):
                    if line.startswith('*NODE'):
                        if not line.startswith('*NODE PRINT') and not line.startswith('*NODE FILE'):
                            nset_name = line.split('=')[1]
                            nset_list[nset_name] = []
                            current_keyword = 'NODE'
                        else:
                            current_keyword = None
                    elif line.startswith('*ELEMENT'):
                        if not line.startswith('*ELEMENT PRINT') and not line.startswith('*ELEMENT FILE'):
                            parts = line.split(',')
                            for part in parts:
                                if part.startswith('TYPE='):
                                    element_type = part.split('=')[1]
                                elif part.startswith('ELSET='):
                                    elset_name = part.split('=')[1]
                            elset_list[elset_name] = []
                            current_keyword = 'ELEMENT'
                        else:
                            current_keyword = None
                    elif line.startswith('*MATERIAL'):
                        material_name = line.split('=')[1]
                        current_keyword = 'MATERIAL'
                    elif line.startswith('*ELASTIC'):
                        current_keyword = 'ELASTIC'
                    elif line.startswith('*DENSITY'):
                        current_keyword = 'DENSITY'
                    elif line.startswith('*SOLID'):
                        parts = line.split(',')
                        for part in parts:
                            if part.startswith('ELSET='):
                                elset_name = part.split('=')[1]
                            elif part.startswith('MATERIAL='):
                                material_name = part.split('=')[1]
                        if material_name not in elmat:
                            elmat[material_name] = []
                        for elem_id in elset_list[elset_name]:
                            elmat[material_name].extend([elem_id])
                    elif line.startswith('*BEAM'):
                        parts = line.split(',')
                        for part in parts:
                            if part.startswith('SECTION='):
                                section_type = part.split('=')[1]
                            elif part.startswith('ELSET='):
                                elset_name = part.split('=')[1]
                            elif part.startswith('MATERIAL='):
                                material_name = part.split('=')[1]
                        if material_name not in elmat:
                            elmat[material_name] = []
                        current_keyword = 'BEAM'
                    elif line.startswith('*SHELL'):
                        parts = line.split(',')
                        for part in parts:
                            if part.startswith('ELSET='):
                                elset_name = part.split('=')[1]
                            elif part.startswith('MATERIAL='):
                                material_name = part.split('=')[1]
                        if material_name not in elmat:
                            elmat[material_name] = []
                        current_keyword = 'SHELL'
                        section_type = 'SHELL'
                    elif line.startswith('*ELSET'):
                        elset_name = line.split('=')[1]
                        elset_list[elset_name] = []
                        current_keyword = 'ELSET'
                    elif line.startswith('*NSET'):
                        nset_name = line.split('=')[1]
                        nset_list[nset_name] = []
                        current_keyword = 'NSET'
                    elif line.startswith('*BOUNDARY'):
                        current_keyword = 'BOUNDARY'
                    else:
                        current_keyword = None
                    continue

                if current_keyword == 'NODE' and line:
                    line.rstrip(',')
                    node_data = line.split(',')
                    node_data = [s.strip() for s in node_data]
                    node_id = int(node_data[0])
                    x = float(node_data[1])
                    y = float(node_data[2])
                    z = float(node_data[3])
                    nset_list[nset_name].extend([node_id])
                    coordinates_list[node_id] = [x, y, z]
                elif current_keyword == 'ELEMENT' and line:
                    line = line.rstrip(',')
                    element_data = line.split(',')
                    element_data = [s.strip() for s in element_data]
                    element_id = int(element_data[0])
                    node_ids = list(map(int, element_data[1:]))
                    connections_list[element_id] = node_ids
                    element_types[element_id] = element_type.strip()
                    elset_list[elset_name].extend([element_id])
                elif current_keyword == 'ELSET' and line:
                    element_data = line.split(',')
                    element_data = [s.strip() for s in element_data]
                    if len(element_data) > 1:
                        element_ids = list(map(int, element_data[0:]))
                        elset_list[elset_name].extend(element_ids)
                    else:
                        element_id = int(element_data[0])
                        elset_list[elset_name].extend([element_id])
                elif current_keyword == 'NSET' and line:
                    node_data = line.split(',')
                    node_data = [s.strip() for s in node_data]
                    if len(node_data) > 1:
                        node_ids = list(map(int, node_data[0:]))
                        nset_list[nset_name].extend(node_ids)
                    else:
                        node_id = int(node_data[0])
                        nset_list[nset_name].extend([node_id])
                elif current_keyword == 'ELASTIC':
                    material_data = line.split(',')
                    material_data = [s.strip() for s in material_data]
                    material_elastic[material_name] = float(material_data[0])
                    material_poisson[material_name] = float(material_data[1])
                elif current_keyword == 'DENSITY':
                    material_data = line.split(',')
                    material_data = [s.strip() for s in material_data]
                    material_density[material_name] = float(material_data[0])
                elif current_keyword == 'BEAM':
                    beam_data = line.split(',')
                    beam_data = [s.strip() for s in beam_data]
                    for elem_id in elset_list[elset_name]:
                        if section_type.strip() == 'RECT':
                            section_types[elem_id] = 'RECT'
                            if len(beam_data) == 2:
                                beam_rect[elem_id] = [float(beam_data[0]), float(beam_data[1])]
                        elif section_type.strip() == 'BOX':
                            section_types[elem_id] = 'BOX'
                            if len(beam_data) == 6:
                                beam_box[elem_id] = [float(beam_data[0]), float(beam_data[1]), float(beam_data[2]), \
                                                     float(beam_data[3]), float(beam_data[4]), float(beam_data[5])]
                        elmat[material_name].extend([elem_id])
                elif current_keyword == 'SHELL':
                    shell_data = line.split(',')
                    shell_data = [s.strip() for s in shell_data]
                    for elem_id in elset_list[elset_name]:
                        section_types[elem_id] = 'SHELL'
                        shell_thickness[elem_id] = float(shell_data[0])
                        elmat[material_name].extend([elem_id])
                elif current_keyword == 'BOUNDARY' and line:
                    boundary_data = line.split(',')
                    boundary_data = [s.strip() for s in boundary_data]
                    try:
                        boundary_point = int(boundary_data[0])
                        self.set_boundary_point(boundary_point,dofs=boundary_data[len(boundary_data[0]):])
                    except valueError:
                        boundary_nset = boundary_data[0]
                        for boundary_point in nset_list[boundary_nset]:
                            self.set_boundary_point(boundary_point,dofs=boundary_data[len(boundary_data[0]):])

            # Define elements and points
            self.set_npoin(len(coordinates_list))
            self.set_nelem(len(connections_list))
            for node_id in coordinates_list:
                self.set_point_coord(node_id-1, coordinates_list[node_id])
            for elem_id in connections_list:
                connections_list[elem_id] = [connections_list[elem_id][i]-1 for i in range(len(connections_list[elem_id]))]
                self.set_element_type(elem_id-1, element_types[elem_id])
                self.set_element_npoin(elem_id-1, len(connections_list[elem_id]))
                self.set_element_point(elem_id-1, connections_list[elem_id])
                if section_types[elem_id] == 'RECT':
                    rect_values = beam_rect[elem_id]
                    self.set_beam_rect(elem_id-1, rect_values[0], rect_values[1])
                elif section_types[elem_id] == 'BOX':
                    box_values = beam_box[elem_id]
                    self.set_beam_box(elem_id-1, box_values[0], box_values[1], box_values[2], \
                                                 box_values[3], box_values[4], box_values[5])
                elif section_types[elem_id] == 'SHELL':
                    self.set_shell_thickness(elem_id-1, shell_thickness[elem_id])

            # Assign materials
            for material in elmat:
                young = material_elastic[material]
                poisson = material_poisson[material]
                density = material_density[material]
                for elem_id in elmat[material]:
                    self.set_young_modulus(elem_id-1, young)
                    self.set_poisson_coef(elem_id-1, poisson)
                    self.set_density(elem_id-1, density)

    """
    [3]  update_strength_factor

         Updates strength factor to track internally
    """
    def update_strength_factor(self, strength_factor):
        for ielem in range(self.nelem):
            self.strength_factor[ielem] = strength_factor[ielem]


    """
    [6]  set_icase

         For sparse grid runs, sets a class state icase so it does not get passed as an function argument
    """
    def set_icase(self, icase):
        self.icase = icase


    """
    [7]  set_ncase

         Update number of case loads
    """
    def set_ncase(self, ncase):
        self.ncase = ncase


    """
    [8]  set_objective_type

         Sets objective type.
              obj_type == -1 == 'basic'               <-- No uncertainty. Standard run.
                       ==  0 == 'neutral'             <-- Risk Neutral integration of uncertainty.
                       ==  1 == 'cvar'                <-- Conditional Value at Risk.
                       ==  2 == 'neutral+cvar'        <-- Risk Neutral + Conditional Value at Risk.
                       ==  3 == 'mean+semideviation'  <-- Mean + Semi Deviation
                       ==  4 == 'entropic_risk'       <-- Entropic Risk
    """
    def set_objective_type(self, obj_type):
        if obj_type == 'basic' or obj_type == -1:
            self.obj_type = -1
        elif obj_type == 'neutral' or obj_type == 0:
            self.obj_type = 0
            self.is_uncertain = True
        elif obj_type == 'cvar' or obj_type == 1:
            self.obj_type = 1
            self.is_uncertain = True
        elif obj_type == 'neutral+cvar' or obj_type == 2:
            self.obj_type = 2
            self.is_uncertain = True
        elif obj_type == 'mean+semideviation' or obj_type == 3:
            print(f"Mean+Semideviation not implemented. Using risk neutral instead.")
            self.obj_type = 0
            self.is_uncertain = True
        elif obj_type == 'entropic_risk' or obj_type == 4:
            print(f"Entropic Risk not implemented. Using risk neutral instead.")
            self.obj_type = 0
            self.is_uncertain = True
        else:
            self.obj_type = -1

    """
    [9]  set_integration_order

         Initializes integration weights and values
    """
    def set_integration_order(self, gauss_order, random_dimension):
        self.integ = integrator(gauss_order, 'hypercube', random_dimension)
        self.nsample = self.integ.integ_terms
        self.random_dimension = random_dimension
        self.random_factor = np.zeros((self.nsample, self.random_dimension), dtype = np.float64)


    """
    [10]  set_cvar_beta

         Sets beta value for Conditional Value at Risk runs
    """
    def set_cvar_beta(self, cvar_beta):
        self.cvar_beta = cvar_beta


    """
    [11]  set_risk_lambda

         Sets lambda coefficient for neutral+cvar runs.
         Objective will be computed as J = lambda*neutral_obj + (1-lambda)*cvar_obj
    """
    def set_risk_lambda(self, risk_lambda):
        self.risk_lambda = risk_lambda


    """
    [12]  set_nproc

         Set number of processors used for parallel executions.
         nproc_ccx   <-- Number of processors for Calculix solves
         nproc_cases <-- Number of processors for simultaneous Calculix solves
    """
    def set_nproc(self, nproc_ccx, nproc_cases):
        self.nproc_ccx = nproc_ccx
        os.environ["OMP_NUM_THREADS"] = str(nproc_ccx)
        os.environ["NUMBER_OF_CPUS"] = str(nproc_ccx)
        self.nproc_cases = nproc_cases


    """
    [13]  set_npoin

         Sets number of points in mesh
    """
    def set_npoin(self, npoin):
        self.npoin = npoin
        self.point_coord = [[0.0, 0.0, 0.0] for i in range(npoin)]


    """
    [14]  set_nelem

         Sets number of elements in mesh
    """
    def set_nelem(self, nelem):
        self.nelem = nelem
        self.element = [calculix_element() for i in range(nelem)]
        for ielem in range(nelem):
            self.element[ielem].forward_unkno = []
            self.element[ielem].adjoint_unkno = []
            self.element[ielem].forward_hessvec_unkno = []
            self.element[ielem].adjoint_hessvec_unkno = []


    """
    [15]  set_nsensor

         Sets number of sensors in digital twin and initializes a kd-tree to place them after.
    """
    def set_nsensor(self, nsensor):
        self.nsensor = nsensor
        self.sensor_point = np.zeros(nsensor, dtype=np.int32)
        self.sensor_unkno_targ = np.zeros((self.ncase, self.ntime, nsensor, 3), dtype=np.float64)
        self.sensor_unkno_curr = np.zeros((self.ncase, self.nsample, self.ntime, nsensor, 3), dtype=np.float64)
        self.point_tree = KDTree(self.point_coord, leafsize=100)


    """
    [16]  set_element_type

         Sets the Calculix element type of element ielem.
         Types supported: 'C3D4', 'C3D6', 'C3D8', 'C3D10', 'C3D15', 'C3D20', 'CPS3',
                          'T3D2', 'B32R', 'B32', 'B31', 'B31R', 'S3, 'S6'
    """
    def set_element_type(self, ielem, element_type):
        self.element[ielem].set_element_type(element_type)


    """
    [17]  set_element_npoin

         Sets the number of points in element ielem
    """
    def set_element_npoin(self, ielem, npoin):
        self.element[ielem].set_npoin(npoin)


    """
    [18]  set_element_point

         Sets the ids of the points in element ielem
    """
    def set_element_point(self, ielem, point):
        self.element[ielem].set_point(point)


    """
    [19]  set_young_modulus

         Sets the Young modulus of element ielem
    """
    def set_young_modulus(self, ielem, young_modulus):
        self.element[ielem].set_young_modulus(young_modulus)


    """
    [20]  set_poisson_coef

         Sets the Poisson coefficient of element ielem
    """
    def set_poisson_coef(self, ielem, poisson_coef):
        self.element[ielem].set_poisson_coef(poisson_coef)


    """
    [21]  set_density

         Sets the density of element ielem
    """
    def set_density(self, ielem, density):
        self.element[ielem].set_density(density)


    """
    [22]  set_section_type

         Sets the section type of element ielem.
         Types supported: 'SOLID', 'RECT', 'BOX', 'PIPE', 'SHELL'
    """
    def set_section_type(self, ielem, section_type):
        self.element[ielem].set_section_type(section_type)


    """
    [23]  set_beam_rect

         Sets the section parameters of a beam with 'RECT' section
    """
    def set_beam_rect(self, ielem, width, height):
        self.element[ielem].set_beam_rect(width, height)


    """
    [24]  set_beam_box

         Sets the section parameters of a beam with 'BOX' section
    """
    def set_beam_box(self, ielem, width, height, t1, t2, t3, t4):
        self.element[ielem].set_beam_box(width, height, t1, t2, t3, t4)


    """
    [25]  set_shell_thickness

         Sets the thickness of a 'SHELL' element
    """
    def set_shell_thickness(self, ielem, thickness):
        self.element[ielem].set_shell_thickness(thickness)


    """
    [26]  set_point_coord

         Sets the coordinates of point ipoin
    """
    def set_point_coord(self, ipoin, coord):
        self.point_coord[ipoin] = coord


    """
    [27]  set_boundary_point

         Sets point ipoin as a boundary point
    """
    def set_boundary_point(self, ipoin, boundary_type=6, dofs='1,6'):
        if ipoin not in self.boundary_type:
            self.boundary_type[ipoin] = {}
            self.boundary_dofs[ipoin] = {}
        self.boundary_type[ipoin] = boundary_type
        self.boundary_dofs[ipoin] = dofs


    """
    [28]  set_sensor_location

         Sets the location of sensor isensor.
         Uses previously defined kd-tree to locate a mesh point near the given location to place them in.
    """
    def add_sensor_location(self, isensor, x, y, z):
        nquery = 3
        dist, point_id = self.point_tree.query([x, y, z], k=nquery)
        found_nn = False
        for iquery in range(nquery):
            if point_id[iquery] not in self.sensor_point:
                self.set_sensor(isensor, point_id[iquery])
                found_nn = True
                break
        if not found_nn:
            raise ValueError(f"Error in add_sensor: too many sensor points with no unique FEM points near.")


    """
    [29]  set_sensor

         Directly assigns mesh point ipoin as sensor isensor.
    """
    def set_sensor(self, isensor, ipoin):
        self.sensor_point[isensor] = ipoin


    """
    [30]  add_sensor_measurement

         Sets the target displacements of a sensor.
    """
    def add_sensor_measurement(self, icase, itime, isensor, u, v, w):
        self.sensor_unkno_targ[icase][itime][isensor][0] = u
        self.sensor_unkno_targ[icase][itime][isensor][1] = v
        self.sensor_unkno_targ[icase][itime][isensor][2] = w


    """
    [31]  add_control_to_vtk

         Adds given control array to vtk output, located in file named filename
    """
    def add_control_to_vtk(self, filename, control):
        if self.ntime <= 1:
            with open(filename.strip() + '.vtk', 'a') as fu:
                fu.write('\n')
                fu.write('CELL_DATA ' + str(self.nelem) + '\n')
                fu.write('SCALARS strf double 1\n')
                fu.write('LOOKUP_TABLE my_table\n')
                for icontrol in range(self.nelem):
                    fu.write(f"{control[icontrol]:16.8E}\n")
        else:
            ndigit = len(str(self.ntime))
            for itime in range(self.ntime):
                with open(filename.strip() + '.' + f"{itime+1:0{ndigit}}" + '.vtk', 'a') as fu:
                    fu.write('\n')
                    fu.write('CELL_DATA ' + str(self.nelem) + '\n')
                    fu.write('SCALARS strf double 1\n')
                    fu.write('LOOKUP_TABLE my_table\n')
                    for icontrol in range(self.nelem):
                        fu.write(f"{control[icontrol]:16.8E}\n")



    """
    [32]  add_grad_to_vtk

         Adds given gradient array to vtk output, located in file named filename
    """
    def add_grad_to_vtk(self, filename, obj_grad, smooth=True):
        if self.ntime <= 1:
            with open(filename.strip() + '.vtk', 'a') as fu:
                fu.write('\n')
                if smooth:
                    fu.write('SCALARS gradient_smooth double 1\n')
                else:
                    fu.write('CELL_DATA ' + str(self.nelem) + '\n')
                    fu.write('SCALARS gradient double 1\n')
                fu.write('LOOKUP_TABLE my_table\n')
                for ielem in range(self.nelem):
                    fu.write('{:16.8E}\n'.format(obj_grad[ielem]))
        else:
            ndigit = len(str(self.ntime))
            for itime in range(self.ntime):
                with open(filename.strip() + '.' + f"{itime+1:0{ndigit}}" + '.vtk', 'a') as fu:
                    fu.write('\n')
                    if smooth:
                        fu.write('SCALARS gradient_smooth double 1\n')
                    else:
                        fu.write('CELL_DATA ' + str(self.nelem) + '\n')
                        fu.write('SCALARS gradient double 1\n')
                    fu.write('LOOKUP_TABLE my_table\n')
                    for ielem in range(self.nelem):
                        fu.write('{:16.8E}\n'.format(obj_grad[ielem]))


    """
    [33]  generate_sensor_diff_vtk

         Generates a vtk file with the sensor points with the difference between the current and target displacements
    """
    def generate_sensor_diff_vtk(self, icase, isample, filename, run_type="optim"):
        ndigit = len(str(self.ntime))
        for itime in range(self.ntime):
            if self.run_type == 'dynamic':
                filename_vtk = filename.strip() + f"_sensor_diff.{itime+1:0{ndigit}}.vtk"
            else:
                filename_vtk = filename.strip() + "_sensor_diff.vtk"
            with open(filename_vtk, 'w') as fu:
                fu.write('# vtk DataFile Version 2.0\n')
                fu.write('Converted Feelast File : Diff_Measuring Point Info\n')
                fu.write('ASCII\n')
                fu.write('DATASET UNSTRUCTURED_GRID\n')
                fu.write('\nPOINTS {} float\n'.format(self.nsensor))
                for isensor in range(self.nsensor):
                    ip = self.sensor_point[isensor]
                    fu.write(f"{self.point_coord[ip][0]:16.8E} {self.point_coord[ip][1]:16.8E} {self.point_coord[ip][2]:16.8E}\n")

                fu.write('\nCELLS {} {}\n'.format(self.nsensor, 2 * self.nsensor))
                for isensor in range(self.nsensor):
                    fu.write(f"1 {isensor}\n")

                fu.write('\nCELL_TYPES {}\n'.format(self.nsensor))
                for isensor in range(self.nsensor):
                    fu.write('1\n')

                fu.write('\nPOINT_DATA {}\n'.format(self.nsensor))
                fu.write('SCALARS diff_meas_point_nr float 1\n')
                fu.write('LOOKUP_TABLE default\n')
                for isensor in range(1, self.nsensor + 1):
                    fu.write('{:16.8E}\n'.format(float(isensor)))

                if run_type.strip() == 'target':
                    fu.write('SCALARS diff_meas_value float 1\n')
                    fu.write('LOOKUP_TABLE default\n')
                    for isensor in range(self.nsensor):
                        um = self.sensor_unkno_targ[icase][itime][isensor][0]
                        vm = self.sensor_unkno_targ[icase][itime][isensor][1]
                        wm = self.sensor_unkno_targ[icase][itime][isensor][2]
                        diff = (um ** 2 + vm ** 2 + wm ** 2) ** 0.5
                        fu.write('{:16.8E}\n'.format(diff))
                    fu.write('VECTORS diff_meas_displac float\n')
                    for isensor in range(self.nsensor):
                        um = self.sensor_unkno_targ[icase][itime][isensor][0]
                        vm = self.sensor_unkno_targ[icase][itime][isensor][1]
                        wm = self.sensor_unkno_targ[icase][itime][isensor][2]
                        fu.write('{:16.8E} {:16.8E} {:16.8E}\n'.format(um, vm, wm))
                else:
                    fu.write('SCALARS diff_meas_value float 1\n')
                    fu.write('LOOKUP_TABLE default\n')
                    for isensor in range(self.nsensor):
                        u = self.sensor_unkno_curr[icase][itime][isample][isensor][0]
                        v = self.sensor_unkno_curr[icase][itime][isample][isensor][1]
                        w = self.sensor_unkno_curr[icase][itime][isample][isensor][2]
                        um = self.sensor_unkno_targ[icase][itime][isensor][0]
                        vm = self.sensor_unkno_targ[icase][itime][isensor][1]
                        wm = self.sensor_unkno_targ[icase][itime][isensor][2]
                        du = u - um
                        dv = v - vm
                        dw = w - wm
                        diff = (du ** 2 + dv ** 2 + dw ** 2) ** 0.5
                        fu.write('{:16.8E}\n'.format(diff))
                    fu.write('VECTORS diff_meas_displac float\n')
                    for isensor in range(self.nsensor):
                        u = self.sensor_unkno_curr[icase][itime][isample][isensor][0]
                        v = self.sensor_unkno_curr[icase][itime][isample][isensor][1]
                        w = self.sensor_unkno_curr[icase][itime][isample][isensor][2]
                        um = self.sensor_unkno_targ[icase][itime][isensor][0]
                        vm = self.sensor_unkno_targ[icase][itime][isensor][1]
                        wm = self.sensor_unkno_targ[icase][itime][isensor][2]
                        du = u - um
                        dv = v - vm
                        dw = w - wm
                        fu.write('{:16.8E} {:16.8E} {:16.8E}\n'.format(du, dv, dw))


    """
    [34]  compute_stiffness

         Runs a calculix solver for each element to retrieve its stiffness matrix and store it
    """
    def compute_stiffness(self):
        os.makedirs('stiffness', exist_ok=True)
        # Compute stiffnesses in parallel
        with ProcessPoolExecutor(max_workers=np.max([self.nproc_cases,self.nproc_ccx])) as executor:
            run_futures = [executor.submit(self.compute_stiffness_serial, ielem) for ielem in range(self.nelem)]
            for future in as_completed(run_futures):
                try:
                    ielem, stiffness = future.result()  # This will raise any exceptions that occurred
                    self.element[ielem].stiffness = stiffness
                except Exception as exc:
                    print(f'Exception in compute_stiffness_serial: {exc}')
        os.system('rm -r stiffness')


    """
    [35]  compute_stiffness_serial

         Computes the stiffnes for element ielem
    """
    def compute_stiffness_serial(self, ielem):
        if ielem%250 == 0 or ielem == self.nelem-1: print(f"computing stiffness of element {ielem}")
        if self.element[ielem].element_type.strip() == 'B32R' or self.element[ielem].element_type.strip() == 'B32':
            n = 60
            self.element[ielem].stiffness = np.zeros((n,n), dtype=np.float64)
        elif self.element[ielem].element_type.strip() == 'B31' or self.element[ielem].element_type.strip() == 'B31R':
            n = 24
            self.element[ielem].stiffness = np.zeros((n,n), dtype=np.float64)
        elif self.element[ielem].element_type.strip() == 'S3':
            n = 18
            self.element[ielem].stiffness = np.zeros((n,n), dtype=np.float64)
        elif self.element[ielem].element_type.strip() == 'S6':
            n = 45
            self.element[ielem].stiffness = np.zeros((n,n), dtype=np.float64)
        elif self.element[ielem].element_type.strip() == 'C3D8':
            n =24
            self.element[ielem].stiffness = np.zeros((n,n), dtype=np.float64)
        elif self.element[ielem].element_type.strip() == 'C3D6':
            n = 18
            self.element[ielem].stiffness = np.zeros((n,n), dtype=np.float64)
        elif self.element[ielem].element_type.strip() == 'C3D4':
            n = 12
            self.element[ielem].stiffness = np.zeros((n,n), dtype=np.float64)
        filename = 'stiffness/calculix_stiffness_' + str(ielem)
        with open(filename + '.inp', 'w') as fu:
            fu.write('**                            \n')
            fu.write('** Calculix stiffness problem \n')
            fu.write('**                            \n')
            fu.write('*NODE,NSET=NALL\n')
            for ipoin in range(self.element[ielem].npoin):
                ip = self.element[ielem].point[ipoin]
                fu.write(f"{ipoin+1}, ")
                fu.write(f"{self.point_coord[ip][0]:16.8E}, ")
                fu.write(f"{self.point_coord[ip][1]:16.8E}, ")
                fu.write(f"{self.point_coord[ip][2]:16.8E}\n")
            fu.write(f"*ELEMENT,TYPE={self.element[ielem].element_type},ELSET=EALL\n")
            fu.write("1,")
            for ipoin in range(self.element[ielem].npoin):
                fu.write(f"{ipoin+1},")
            fu.write('\n')
            fu.write('*MATERIAL,NAME=MAT\n')
            fu.write('*ELASTIC\n')
            fu.write(f"{self.element[ielem].young_modulus:16.8E}, {self.element[ielem].poisson_coef:16.8E}\n")
            fu.write('*DENSITY\n')
            fu.write(f"{self.element[ielem].density:16.8E}\n")
            if self.element[ielem].section_type.strip() == 'SOLID':
                fu.write('*SOLID SECTION,ELSET=EALL,MATERIAL=MAT\n')
            elif self.element[ielem].section_type.strip() == 'RECT':
                fu.write('*BEAM SECTION,ELSET=EALL,MATERIAL=MAT,SECTION=RECT\n')
                fu.write(f"{self.element[ielem].beam_width:16.8E}, {self.element[ielem].beam_height:16.8E}\n")
                first_normal = self.get_beam_first_normal(ielem)
                fu.write(f"{first_normal[0]:16.8E}, {first_normal[1]:16.8E}, {first_normal[2]:16.8E}\n")
            elif self.element[ielem].section_type.strip() == 'BOX':
                fu.write('*BEAM SECTION,ELSET=EALL,MATERIAL=MAT,SECTION=BOX\n')
                fu.write(f"{self.element[ielem].beam_width:16.8E}, {self.element[ielem].beam_height:16.8E},")
                fu.write(f"{self.element[ielem].beam_t1:16.8E}, {self.element[ielem].beam_t2:16.8E},")
                fu.write(f"{self.element[ielem].beam_t3:16.8E}, {self.element[ielem].beam_t4:16.8E}\n")
                first_normal = self.get_beam_first_normal(ielem)
                fu.write(f"{first_normal[0]:16.8E}, {first_normal[1]:16.8E}, {first_normal[2]:16.8E}\n")
            elif self.element[ielem].section_type.strip() == 'SHELL':
                fu.write('*SHELL SECTION,ELSET=EALL,MATERIAL=MAT\n')
                fu.write(f"{self.element[ielem].shell_thickness:16.8E}\n")
            else:
                print('Element section type', self.element[ielem].section_type.strip(), 'unrecognized')
            fu.write('*STEP\n')
            fu.write('*FREQUENCY,SOLVER=MATRIXSTORAGE\n')
            fu.write('*END STEP\n')
        runstr = 'ccx ' + filename + ' > ' + filename + '.out'
        os.system(runstr)
        if self.element[ielem].element_type.strip() == 'B32R' or self.element[ielem].element_type.strip() == 'B32':
            n = 60
            stiffness = np.zeros((n,n), dtype=np.float64)
        elif self.element[ielem].element_type.strip() == 'B31' or self.element[ielem].element_type.strip() == 'B31R':
            n = 24
            stiffness = np.zeros((n,n), dtype=np.float64)
        elif self.element[ielem].element_type.strip() == 'S3':
            n = 18
            stiffness = np.zeros((n,n), dtype=np.float64)
        elif self.element[ielem].element_type.strip() == 'S6':
            n = 45
            stiffness = np.zeros((n,n), dtype=np.float64)
        elif self.element[ielem].element_type.strip() == 'C3D8':
            n =24
            stiffness = np.zeros((n,n), dtype=np.float64)
        elif self.element[ielem].element_type.strip() == 'C3D6':
            n = 18
            stiffness = np.zeros((n,n), dtype=np.float64)
        elif self.element[ielem].element_type.strip() == 'C3D4':
            n = 12
            stiffness = np.zeros((n,n), dtype=np.float64)
        with open(filename + '.sti', 'r') as fu:
            for line in fu:
                data = line.split()
                if len(data) == 3:
                    row, col, val = int(data[0]), int(data[1]), float(data[2])
                    if row <= n and col <= n:
                        stiffness[row-1][col-1] = -val
                        stiffness[col-1][row-1] = -val
                else:
                    print('Error reading stiffness matrix')
                    raise ValueError('Error reading stiffness matrix')
        runstr = 'rm ' + filename + '.*'
        os.system(runstr)

        return ielem, stiffness


    """
    [36]  get_beam_first_normal

         Selects the first normal direction for the section of a beam element
    """
    def get_beam_first_normal(self, ielem):
        ip1 = self.element[ielem].point[0]
        ip2 = self.element[ielem].point[-1]
        dx = self.point_coord[ip2][0] - self.point_coord[ip1][0]
        dy = self.point_coord[ip2][1] - self.point_coord[ip1][1]
        dz = self.point_coord[ip2][2] - self.point_coord[ip1][2]
        dl = np.sqrt(dx*dx + dy*dy + dz*dz)
        t1 = dx / dl
        t2 = dy / dl
        t3 = dz / dl
        EPS = 1e-6  # Define EPS value here
        first_normal = np.zeros(3, dtype=np.float64)
        if abs(t3) < EPS:
            first_normal[0] = 0.0
            first_normal[1] = 0.0
            first_normal[2] = 1.0
        elif abs(t1) < EPS:
            first_normal[0] = -1.0
            first_normal[1] = 0.0
            first_normal[2] = 0.0
        else:
            dl = np.sqrt(dx*dx + dz*dz)
            first_normal[0] = -dz / dl
            first_normal[1] = 0.0
            first_normal[2] = dx / dl
        return first_normal


    """
    [36]  compute_mass

         Computes the volume of each element and saves them as mass.
         This is useful in the case that an external integration is necessary.
    """
    def compute_mass(self):
        self.mass = np.zeros(self.nelem, dtype=np.float64)
        for ielem in range(self.nelem):
            if self.element[ielem].element_type.strip() in ['B32', 'B32R', 'B31', 'B31R']:
                ip1 = self.element[ielem].point[0]
                ip3 = self.element[ielem].point[-1]
                x13 = self.point_coord[ip3][0] - self.point_coord[ip1][0]
                y13 = self.point_coord[ip3][1] - self.point_coord[ip1][1]
                z13 = self.point_coord[ip3][2] - self.point_coord[ip1][2]
                self.mass[ielem] = np.sqrt(x13 ** 2 + y13 ** 2 + z13 ** 2)
                if self.element[ielem].section_type.strip() == 'RECT':
                    width = self.element[ielem].beam_width
                    height = self.element[ielem].beam_height
                    self.mass[ielem] *= width * height
                elif self.element[ielem].section_type.strip() == 'BOX':
                    width = self.element[ielem].beam_width
                    height = self.element[ielem].beam_height
                    t1 = self.element[ielem].beam_t1
                    t2 = self.element[ielem].beam_t2
                    t3 = self.element[ielem].beam_t3
                    t4 = self.element[ielem].beam_t4
                    outter_area = width * height
                    inner_area = (width - t1 - t3) * (height - t2 - t4)
                    self.mass[ielem] *= (outter_area - inner_area)
            elif self.element[ielem].element_type.strip() in ['S6', 'S3']:
                ip1 = self.element[ielem].point[0]
                ip2 = self.element[ielem].point[1]
                ip3 = self.element[ielem].point[2]
                x12 = self.point_coord[ip2][0] - self.point_coord[ip1][0]
                y12 = self.point_coord[ip2][1] - self.point_coord[ip1][1]
                z12 = self.point_coord[ip2][2] - self.point_coord[ip1][2]
                x13 = self.point_coord[ip3][0] - self.point_coord[ip1][0]
                y13 = self.point_coord[ip3][1] - self.point_coord[ip1][1]
                z13 = self.point_coord[ip3][2] - self.point_coord[ip1][2]
                self.mass[ielem] = 0.5 * np.sqrt((y12 * z13 - z12 * y13) ** 2 +
                                                 (z12 * x13 - x12 * z13) ** 2 +
                                                 (x12 * y13 - y12 * x13) ** 2)
                self.mass[ielem] *= self.element[ielem].shell_thickness
            elif self.element[ielem].element_type.strip() == 'C3D4':
                ip1 = self.element[ielem].point[0]
                ip2 = self.element[ielem].point[1]
                ip3 = self.element[ielem].point[2]
                ip4 = self.element[ielem].point[3]
                vol1 = ((self.point_coord[ip4][0] - self.point_coord[ip1][0]) *
                        ((self.point_coord[ip2][1] - self.point_coord[ip1][1]) *
                         (self.point_coord[ip3][2] - self.point_coord[ip1][2]) -
                         (self.point_coord[ip2][2] - self.point_coord[ip1][2]) *
                         (self.point_coord[ip3][1] - self.point_coord[ip1][1])) +
                        (self.point_coord[ip4][1] - self.point_coord[ip1][1]) *
                        ((self.point_coord[ip2][2] - self.point_coord[ip1][2]) *
                         (self.point_coord[ip3][0] - self.point_coord[ip1][0]) -
                         (self.point_coord[ip2][0] - self.point_coord[ip1][0]) *
                         (self.point_coord[ip3][2] - self.point_coord[ip1][2])) +
                        (self.point_coord[ip4][2] - self.point_coord[ip1][2]) *
                        ((self.point_coord[ip2][0] - self.point_coord[ip1][0]) *
                         (self.point_coord[ip3][1] - self.point_coord[ip1][1]) -
                         (self.point_coord[ip2][1] - self.point_coord[ip1][1]) *
                         (self.point_coord[ip3][0] - self.point_coord[ip1][0]))) / 6.0
                self.mass[ielem] = abs(vol1)
            elif self.element[ielem].element_type.strip() == 'C3D6':
                ip1 = self.element[ielem].point[0]
                ip2 = self.element[ielem].point[1]
                ip3 = self.element[ielem].point[2]
                ip4 = self.element[ielem].point[3]
                w = np.sqrt((self.point_coord[ip1][0] - self.point_coord[ip4][0]) ** 2 +
                            (self.point_coord[ip1][1] - self.point_coord[ip4][1]) ** 2 +
                            (self.point_coord[ip1][2] - self.point_coord[ip4][2]) ** 2)
                d = np.sqrt((self.point_coord[ip1][0] - self.point_coord[ip3][0]) ** 2 +
                            (self.point_coord[ip1][1] - self.point_coord[ip3][1]) ** 2 +
                            (self.point_coord[ip1][2] - self.point_coord[ip3][2]) ** 2)
                h = np.sqrt((self.point_coord[ip2][0] - self.point_coord[ip3][0]) ** 2 +
                            (self.point_coord[ip2][1] - self.point_coord[ip3][1]) ** 2 +
                            (self.point_coord[ip2][2] - self.point_coord[ip3][2]) ** 2)
                self.mass[ielem] = 0.5 * w * d * h
            elif self.element[ielem].element_type.strip() == 'C3D8':
                ip1 = self.element[ielem].point[0]
                ip2 = self.element[ielem].point[1]
                ip3 = self.element[ielem].point[2]
                ip4 = self.element[ielem].point[5]
                vol1 = ((self.point_coord[ip4][0] - self.point_coord[ip1][0]) *
                        ((self.point_coord[ip2][1] - self.point_coord[ip1][1]) *
                         (self.point_coord[ip3][2] - self.point_coord[ip1][2]) -
                         (self.point_coord[ip2][2] - self.point_coord[ip1][2]) *
                         (self.point_coord[ip3][1] - self.point_coord[ip1][1])) +
                        (self.point_coord[ip4][1] - self.point_coord[ip1][1]) *
                        ((self.point_coord[ip2][2] - self.point_coord[ip1][2]) *
                         (self.point_coord[ip3][0] - self.point_coord[ip1][0]) -
                         (self.point_coord[ip2][0] - self.point_coord[ip1][0]) *
                         (self.point_coord[ip3][2] - self.point_coord[ip1][2])) +
                        (self.point_coord[ip4][2] - self.point_coord[ip1][2]) *
                        ((self.point_coord[ip2][0] - self.point_coord[ip1][0]) *
                         (self.point_coord[ip3][1] - self.point_coord[ip1][1]) -
                         (self.point_coord[ip2][1] - self.point_coord[ip1][1]) *
                         (self.point_coord[ip3][0] - self.point_coord[ip1][0]))) / 6.0
                ip1 = self.element[ielem].point[0]
                ip2 = self.element[ielem].point[2]
                ip3 = self.element[ielem].point[3]
                ip4 = self.element[ielem].point[7]
                vol2 = ((self.point_coord[ip4][0] - self.point_coord[ip1][0]) *
                        ((self.point_coord[ip2][1] - self.point_coord[ip1][1]) *
                         (self.point_coord[ip3][2] - self.point_coord[ip1][2]) -
                         (self.point_coord[ip2][2] - self.point_coord[ip1][2]) *
                         (self.point_coord[ip3][1] - self.point_coord[ip1][1])) +
                        (self.point_coord[ip4][1] - self.point_coord[ip1][1]) *
                        ((self.point_coord[ip2][2] - self.point_coord[ip1][2]) *
                         (self.point_coord[ip3][0] - self.point_coord[ip1][0]) -
                         (self.point_coord[ip2][0] - self.point_coord[ip1][0]) *
                         (self.point_coord[ip3][2] - self.point_coord[ip1][2])) +
                        (self.point_coord[ip4][2] - self.point_coord[ip1][2]) *
                        ((self.point_coord[ip2][0] - self.point_coord[ip1][0]) *
                         (self.point_coord[ip3][1] - self.point_coord[ip1][1]) -
                         (self.point_coord[ip2][1] - self.point_coord[ip1][1]) *
                         (self.point_coord[ip3][0] - self.point_coord[ip1][0]))) / 6.0
                ip1 = self.element[ielem].point[0]
                ip2 = self.element[ielem].point[5]
                ip3 = self.element[ielem].point[7]
                ip4 = self.element[ielem].point[4]
                vol3 = ((self.point_coord[ip4][0] - self.point_coord[ip1][0]) *
                        ((self.point_coord[ip2][1] - self.point_coord[ip1][1]) *
                         (self.point_coord[ip3][2] - self.point_coord[ip1][2]) -
                         (self.point_coord[ip2][2] - self.point_coord[ip1][2]) *
                         (self.point_coord[ip3][1] - self.point_coord[ip1][1])) +
                        (self.point_coord[ip4][1] - self.point_coord[ip1][1]) *
                        ((self.point_coord[ip2][2] - self.point_coord[ip1][2]) *
                         (self.point_coord[ip3][0] - self.point_coord[ip1][0]) -
                         (self.point_coord[ip2][0] - self.point_coord[ip1][0]) *
                         (self.point_coord[ip3][2] - self.point_coord[ip1][2])) +
                        (self.point_coord[ip4][2] - self.point_coord[ip1][2]) *
                        ((self.point_coord[ip2][0] - self.point_coord[ip1][0]) *
                         (self.point_coord[ip3][1] - self.point_coord[ip1][1]) -
                         (self.point_coord[ip2][1] - self.point_coord[ip1][1]) *
                         (self.point_coord[ip3][0] - self.point_coord[ip1][0]))) / 6.0
                ip1 = self.element[ielem].point[2]
                ip2 = self.element[ielem].point[5]
                ip3 = self.element[ielem].point[6]
                ip4 = self.element[ielem].point[7]
                vol4 = ((self.point_coord[ip4][0] - self.point_coord[ip1][0]) *
                        ((self.point_coord[ip2][1] - self.point_coord[ip1][1]) *
                         (self.point_coord[ip3][2] - self.point_coord[ip1][2]) -
                         (self.point_coord[ip2][2] - self.point_coord[ip1][2]) *
                         (self.point_coord[ip3][1] - self.point_coord[ip1][1])) +
                        (self.point_coord[ip4][1] - self.point_coord[ip1][1]) *
                        ((self.point_coord[ip2][2] - self.point_coord[ip1][2]) *
                         (self.point_coord[ip3][0] - self.point_coord[ip1][0]) -
                         (self.point_coord[ip2][0] - self.point_coord[ip1][0]) *
                         (self.point_coord[ip3][2] - self.point_coord[ip1][2])) +
                        (self.point_coord[ip4][2] - self.point_coord[ip1][2]) *
                        ((self.point_coord[ip2][0] - self.point_coord[ip1][0]) *
                         (self.point_coord[ip3][1] - self.point_coord[ip1][1]) -
                         (self.point_coord[ip2][1] - self.point_coord[ip1][1]) *
                         (self.point_coord[ip3][0] - self.point_coord[ip1][0]))) / 6.0
                ip1 = self.element[ielem].point[0]
                ip2 = self.element[ielem].point[5]
                ip3 = self.element[ielem].point[2]
                ip4 = self.element[ielem].point[7]
                vol5 = ((self.point_coord[ip4][0] - self.point_coord[ip1][0]) *
                        ((self.point_coord[ip2][1] - self.point_coord[ip1][1]) *
                         (self.point_coord[ip3][2] - self.point_coord[ip1][2]) -
                         (self.point_coord[ip2][2] - self.point_coord[ip1][2]) *
                         (self.point_coord[ip3][1] - self.point_coord[ip1][1])) +
                        (self.point_coord[ip4][1] - self.point_coord[ip1][1]) *
                        ((self.point_coord[ip2][2] - self.point_coord[ip1][2]) *
                         (self.point_coord[ip3][0] - self.point_coord[ip1][0]) -
                         (self.point_coord[ip2][0] - self.point_coord[ip1][0]) *
                         (self.point_coord[ip3][2] - self.point_coord[ip1][2])) +
                        (self.point_coord[ip4][2] - self.point_coord[ip1][2]) *
                        ((self.point_coord[ip2][0] - self.point_coord[ip1][0]) *
                         (self.point_coord[ip3][1] - self.point_coord[ip1][1]) -
                         (self.point_coord[ip2][1] - self.point_coord[ip1][1]) *
                         (self.point_coord[ip3][0] - self.point_coord[ip1][0]))) / 6.0
                self.mass[ielem] = abs(vol1) + abs(vol2) + abs(vol3) + abs(vol4) + abs(vol5)
            else:
                print('Unrecognized element type for element', ielem)
                self.mass[ielem] = 1.0


    """
    [37]  get_unknown_at_sensors

         Reads the sensor displacements for given .dat file filename for load case icase.
    """
    def get_unknown_at_sensors(self, filename):
        unknown = np.zeros((self.ntime, self.nsensor, 3), dtype=np.float64)
        point_id = np.zeros(self.npoin, dtype=np.int64)
        point_displ = np.zeros((self.ntime, self.npoin, 3), dtype=np.float64)
        with open(filename.strip()) as fu:
            fu.readline()
            itime = -1
            while itime < self.ntime-1:
                itime += 1
                line = fu.readline()
                sline = line.split()
                fu.readline()
                if sline[0] == 'internal':
                    while True:
                        line = fu.readline()
                        # Strip leading and trailing whitespace (including newlines)
                        stripped_line = line.strip()
                        # Check if the line is empty
                        if not stripped_line:
                            itime -= 1
                            break
                elif sline[0] == 'displacements':
                    for ipoin in range(self.npoin):
                        line = fu.readline()
                        parts = line.split()
                        u, v, w = map(float, parts[1:])
                        point_id[ipoin] = int(parts[0])
                        point_displ[itime][ipoin][0] = u
                        point_displ[itime][ipoin][1] = v
                        point_displ[itime][ipoin][2] = w
                    fu.readline()
        for point in point_id:
            for itime in range(self.ntime):
                for isensor in range(self.nsensor):
                    if point-1 == self.sensor_point[isensor]:
                        unknown[itime][isensor][0] = point_displ[itime][point-1][0]
                        unknown[itime][isensor][1] = point_displ[itime][point-1][1]
                        unknown[itime][isensor][2] = point_displ[itime][point-1][2]
        return unknown

    """
    [38]  get_unknowns

         Given a case and sample number, reads the .frd files to retrieve the forward and adjoint unknowns
         for each element. Necessary to compute ut*K*u.
         Needs to do some work to properly place every point variable based on element type.
    """
    def get_unknowns(self, itime, icase, isample, run_type='grad'):
        MAX_PTR = int(5e5)
        ptr = np.zeros(MAX_PTR, dtype=np.int32)

        if run_type == 'grad':
            filename = 'results/calculix_forward_case' + str(icase) + '_sample' + str(isample)
        elif run_type == 'hessvec':
            filename = 'results/calculix_forward_hv_case' + str(icase) + '_sample' + str(isample)
        else:
            raise ValueError("In get_unknowns: wrong run_type")
        with open(filename + '.frd') as fu:
            for _ in range(10):
                fu.readline()
            for _ in range(self.nelem):
                fu.readline()
            _, npoin3d, _ = fu.readline().split()
            npoin3d = int(npoin3d)
            coord3d = np.zeros((3, npoin3d), dtype=np.float64)
            displ = np.zeros((3, npoin3d), dtype=np.float64)
            for ipoin in range(npoin3d):
                line = fu.readline()
                ptr_val = int(line[3:13])
                coord3d[0, ipoin] = float(line[13:25])
                coord3d[1, ipoin] = float(line[25:37])
                coord3d[2, ipoin] = float(line[37:49])
                if ptr_val > MAX_PTR:
                    print("In calculix_interface::get_unknowns: Increase MAX_PTR")
                    raise ValueError("ptr_val exceeds MAX_PTR")
                ptr[ptr_val] = ipoin
            for _ in range(2):
                fu.readline()
            connect3d = np.zeros((20, self.nelem), dtype=np.int32)
            npoin_per_elem = np.zeros(self.nelem, dtype=np.int32)
            for ielem in range(self.nelem):
                parts = fu.readline().split()
                elem_type = int(parts[2])
                npoin = self.get_npoin_from_frd_type(elem_type)
                npoin_per_elem[ielem] = npoin
                if run_type == 'grad':
                    self.element[ielem].forward_unkno = np.zeros(3*npoin, dtype=np.float64)
                elif run_type == 'hessvec':
                    self.element[ielem].forward_hessvec_unkno = np.zeros(3*npoin, dtype=np.float64)
                if npoin == 4:
                    parts = fu.readline().split()
                    connect3d[:4, ielem] = list(map(int, parts[1:]))
                elif npoin == 6:
                    parts = fu.readline().split()
                    connect3d[:6, ielem] = list(map(int, parts[1:]))
                elif npoin == 8:
                    parts = fu.readline().split()
                    connect3d[:8, ielem] = list(map(int, parts[1:]))
                elif npoin == 15:
                    parts = fu.readline().split()
                    connect3d[:10, ielem] = list(map(int, parts[1:]))
                    parts = fu.readline().split()
                    connect3d[10:15, ielem] = list(map(int, parts[1:]))
                elif npoin == 20:
                    parts = fu.readline().split()
                    connect3d[:10, ielem] = list(map(int, parts[1:]))
                    parts = fu.readline().split()
                    connect3d[10:20, ielem] = list(map(int, parts[1:]))

            # Fix connect
            for ielem in range(self.nelem):
                for ipoin in range(npoin_per_elem[ielem]):
                    connect3d[ipoin, ielem] = ptr[connect3d[ipoin, ielem]]

            # Transform connection matrix to have the same order as the stiffness matrix
            # Only necessary for beams
            for ielem in range(self.nelem):
                if npoin_per_elem[ielem] == 20:
                    connect_aux = np.zeros(20, dtype=np.int32)
                    connect_aux[0]  = connect3d[0][ielem]
                    connect_aux[1]  = connect3d[3][ielem]
                    connect_aux[2]  = connect3d[7][ielem]
                    connect_aux[3]  = connect3d[4][ielem]
                    connect_aux[4]  = connect3d[11][ielem]
                    connect_aux[5]  = connect3d[15][ielem]
                    connect_aux[6]  = connect3d[19][ielem]
                    connect_aux[7]  = connect3d[12][ielem]
                    connect_aux[8]  = connect3d[8][ielem]
                    connect_aux[9]  = connect3d[10][ielem]
                    connect_aux[10] = connect3d[18][ielem]
                    connect_aux[11] = connect3d[16][ielem]
                    connect_aux[12] = connect3d[1][ielem]
                    connect_aux[13] = connect3d[2][ielem]
                    connect_aux[14] = connect3d[6][ielem]
                    connect_aux[15] = connect3d[5][ielem]
                    connect_aux[16] = connect3d[9][ielem]
                    connect_aux[17] = connect3d[14][ielem]
                    connect_aux[18] = connect3d[17][ielem]
                    connect_aux[19] = connect3d[13][ielem]
                    connect3d[:20, ielem] = connect_aux
                elif npoin_per_elem[ielem] == 15:
                    connect_aux = np.zeros(15, dtype=np.int32)
                    connect_aux[0]  = connect3d[0][ielem]
                    connect_aux[1]  = connect3d[9][ielem]
                    connect_aux[2]  = connect3d[3][ielem]
                    connect_aux[3]  = connect3d[1][ielem]
                    connect_aux[4]  = connect3d[10][ielem]
                    connect_aux[5]  = connect3d[4][ielem]
                    connect_aux[6]  = connect3d[2][ielem]
                    connect_aux[7]  = connect3d[11][ielem]
                    connect_aux[8]  = connect3d[5][ielem]
                    connect_aux[9]  = connect3d[6][ielem]
                    connect_aux[10] = connect3d[12][ielem]
                    connect_aux[11] = connect3d[7][ielem]
                    connect_aux[12] = connect3d[13][ielem]
                    connect_aux[13] = connect3d[8][ielem]
                    connect_aux[14] = connect3d[14][ielem]
                    connect3d[:15, ielem] = connect_aux

            for _ in range(itime*(8+npoin3d)):
                fu.readline()

            for _ in range(8):
                fu.readline()

            for ipoin in range(npoin3d):
                line = fu.readline()
                displ[0, ipoin] = float(line[13:25])
                displ[1, ipoin] = float(line[25:37])
                displ[2, ipoin] = float(line[37:49])

            # Assign these displacements to the elements
            for ielem in range(self.nelem):
                for ipoin in range(npoin_per_elem[ielem]):
                    ip = connect3d[ipoin, ielem]
                    if run_type == 'grad':
                        self.element[ielem].forward_unkno[3*ipoin:3*(ipoin+1)] = displ[:, ip]
                    elif run_type == 'hessvec':
                        self.element[ielem].forward_hessvec_unkno[3*ipoin:3*(ipoin+1)] = displ[:, ip]

        if run_type == 'grad':
            filename = 'results/calculix_adjoint_case' + str(icase) + '_sample' + str(isample)
        elif run_type == 'hessvec':
            filename = 'results/calculix_adjoint_hv_case' + str(icase) + '_sample' + str(isample)
        with open(filename + '.frd') as fu:
            for _ in range(10):
                fu.readline()
            for _ in range(self.nelem):
                fu.readline()
            _, npoin3d, _ = fu.readline().split()
            npoin3d = int(npoin3d)
            coord3d = np.zeros((3, npoin3d), dtype=np.float64)
            displ = np.zeros((3, npoin3d), dtype=np.float64)
            for ipoin in range(npoin3d):
                line = fu.readline()
                ptr_val = int(line[3:13])
                coord3d[0, ipoin] = float(line[13:25])
                coord3d[1, ipoin] = float(line[25:37])
                coord3d[2, ipoin] = float(line[37:49])
            for _ in range(2):
                fu.readline()
            for ielem in range(self.nelem):
                parts = fu.readline().split()
                elem_type = int(parts[2])
                npoin = self.get_npoin_from_frd_type(elem_type)
                if run_type == 'grad':
                    self.element[ielem].adjoint_unkno = np.zeros(3*npoin, dtype=np.float64)
                elif run_type == 'hessvec':
                    self.element[ielem].adjoint_hessvec_unkno = np.zeros(3*npoin, dtype=np.float64)
                if npoin == 4:
                    parts = fu.readline().split()
                    connect3d[:4, ielem] = list(map(int, parts[1:]))
                elif npoin == 6:
                    parts = fu.readline().split()
                    connect3d[:6, ielem] = list(map(int, parts[1:]))
                elif npoin == 8:
                    parts = fu.readline().split()
                    connect3d[:8, ielem] = list(map(int, parts[1:]))
                elif npoin == 15:
                    parts = fu.readline().split()
                    connect3d[:10, ielem] = list(map(int, parts[1:]))
                    parts = fu.readline().split()
                    connect3d[10:15, ielem] = list(map(int, parts[1:]))
                elif npoin == 20:
                    parts = fu.readline().split()
                    connect3d[:10, ielem] = list(map(int, parts[1:]))
                    parts = fu.readline().split()
                    connect3d[10:20, ielem] = list(map(int, parts[1:]))

            # Fix connect
            for ielem in range(self.nelem):
                for ipoin in range(npoin_per_elem[ielem]):
                    connect3d[ipoin, ielem] = ptr[connect3d[ipoin, ielem]]

            # Transform connection matrix to have the same order as the stiffness matrix
            # Only necessary for beams
            for ielem in range(self.nelem):
                if npoin_per_elem[ielem] == 20:
                    connect_aux = np.zeros(20, dtype=np.int32)
                    connect_aux[0]  = connect3d[0][ielem]
                    connect_aux[1]  = connect3d[3][ielem]
                    connect_aux[2]  = connect3d[7][ielem]
                    connect_aux[3]  = connect3d[4][ielem]
                    connect_aux[4]  = connect3d[11][ielem]
                    connect_aux[5]  = connect3d[15][ielem]
                    connect_aux[6]  = connect3d[19][ielem]
                    connect_aux[7]  = connect3d[12][ielem]
                    connect_aux[8]  = connect3d[8][ielem]
                    connect_aux[9]  = connect3d[10][ielem]
                    connect_aux[10] = connect3d[18][ielem]
                    connect_aux[11] = connect3d[16][ielem]
                    connect_aux[12] = connect3d[1][ielem]
                    connect_aux[13] = connect3d[2][ielem]
                    connect_aux[14] = connect3d[6][ielem]
                    connect_aux[15] = connect3d[5][ielem]
                    connect_aux[16] = connect3d[9][ielem]
                    connect_aux[17] = connect3d[14][ielem]
                    connect_aux[18] = connect3d[17][ielem]
                    connect_aux[19] = connect3d[13][ielem]
                    connect3d[:20, ielem] = connect_aux
                elif npoin_per_elem[ielem] == 15:
                    connect_aux = np.zeros(15, dtype=np.int32)
                    connect_aux[0]  = connect3d[0][ielem]
                    connect_aux[1]  = connect3d[9][ielem]
                    connect_aux[2]  = connect3d[3][ielem]
                    connect_aux[3]  = connect3d[1][ielem]
                    connect_aux[4]  = connect3d[10][ielem]
                    connect_aux[5]  = connect3d[4][ielem]
                    connect_aux[6]  = connect3d[2][ielem]
                    connect_aux[7]  = connect3d[11][ielem]
                    connect_aux[8]  = connect3d[5][ielem]
                    connect_aux[9]  = connect3d[6][ielem]
                    connect_aux[10] = connect3d[12][ielem]
                    connect_aux[11] = connect3d[7][ielem]
                    connect_aux[12] = connect3d[13][ielem]
                    connect_aux[13] = connect3d[8][ielem]
                    connect_aux[14] = connect3d[14][ielem]
                    connect3d[:15, ielem] = connect_aux

            for _ in range((self.ntime-1-itime)*(8+npoin3d)):
                fu.readline()

            for _ in range(8):
                fu.readline()

            for ipoin in range(npoin3d):
                line = fu.readline()
                displ[0, ipoin] = float(line[13:25])
                displ[1, ipoin] = float(line[25:37])
                displ[2, ipoin] = float(line[37:49])

            # Assign these displacements to the elements
            for ielem in range(self.nelem):
                for ipoin in range(npoin_per_elem[ielem]):
                    ip = connect3d[ipoin, ielem]
                    if run_type == 'grad':
                        self.element[ielem].adjoint_unkno[3*ipoin:3*(ipoin+1)] = displ[:, ip]
                    elif run_type == 'hessvec':
                        self.element[ielem].adjoint_hessvec_unkno[3*ipoin:3*(ipoin+1)] = displ[:, ip]

    """
    [39]  get_npoin_from_frd_type

         Given an element type, return the number of point it has, based on Calculix's conventions
    """
    def get_npoin_from_frd_type(self, elem_type):
        if elem_type == 1:
            npoin = 8
        elif elem_type == 2:
            npoin = 6
        elif elem_type == 3:
            npoin = 4
        elif elem_type == 4:
            npoin = 20
        elif elem_type == 5:
            npoin = 15
        else:
            print(f"Unimplemented elem_type: {elem_type}")
            raise ValueError("Unimplemented elem_type")
        return npoin


    """
    [40]  resize_element_array

         Resizes an element array so that values from a 1D or 2D element,
         which Calculix extends to 3D for computations, can be used.
    """
    def resize_element_array(self, ielem, array):
        array_out = np.zeros((self.element[ielem].npoin, 3), dtype=np.float64)

        if array.size == 3*self.element[ielem].npoin:
            for ipoin in range(self.element[ielem].npoin):
                array_out[ipoin][:] = array[3*ipoin:3*(ipoin+1)]
            return array_out

        if array.size == 3*15:
            if self.element[ielem].npoin == 6:
                array_out[0][:] = (array[0:3] + array[9:12] + array[27:30]) / 3.0
                array_out[1][:] = (array[3:6] + array[12:15] + array[30:33]) / 3.0
                array_out[2][:] = (array[6:9] + array[15:18] + array[33:36]) / 3.0
                array_out[3][:] = (array[18:21] + array[36:39]) / 2.0
                array_out[4][:] = (array[21:24] + array[39:42]) / 2.0
                array_out[5][:] = (array[24:27] + array[42:45]) / 2.0
            else:
                raise ValueError("Invalid number of nodes for a 3D wedge element.")
        elif array.size == 3*20:
            if self.element[ielem].npoin == 3:
                array_out[0][:] = (array[0:3] + array[9:12] + array[12:15] +
                                   array[21:24] + array[33:36] + array[36:39] +
                                   array[45:48] + array[57:60])
                array_out[1][:] = (array[24:27] + array[30:33] + array[48:51] +
                                   array[54:57])
                array_out[2][:] = (array[3:6] + array[6:9] + array[15:18] +
                                   array[18:21] + array[27:30] + array[39:42] +
                                   array[42:45] + array[51:54])
            else:
                raise ValueError("Invalid number of nodes for a 3D brick element.")
        return array_out


    """
    [41]  get_objective

         Uses current and target sensor displacements to compute the objective function
         for a given case and sample number
    """
    def get_objective(self, icase, isample):
        obj = 0.0
        for itime in range(self.ntime):
            for isensor in range(self.nsensor):
                u = self.sensor_unkno_curr[icase][itime][isample][isensor][0]
                v = self.sensor_unkno_curr[icase][itime][isample][isensor][1]
                w = self.sensor_unkno_curr[icase][itime][isample][isensor][2]
                um = self.sensor_unkno_targ[icase][itime][isensor][0]
                vm = self.sensor_unkno_targ[icase][itime][isensor][1]
                wm = self.sensor_unkno_targ[icase][itime][isensor][2]
                du = u - um
                dv = v - vm
                dw = w - wm
                dl2 = du**2 + dv**2 + dw**2
                obj += 0.5*dl2
        return obj


    """
    [42]  get_gradient

         Computes grad = ut*K*u for each element. Applies smoothing if necessary.
    """
    def get_gradient(self, icase, isample):
        obj_grad = np.zeros(self.nelem, dtype=np.float64)
        for itime in range(self.ntime):
            self.get_unknowns(itime, icase, isample)
            for ielem in range(self.nelem):
                Ku = np.matmul(self.element[ielem].stiffness, self.element[ielem].forward_unkno)
                vKu = np.dot(self.element[ielem].adjoint_unkno, Ku)
                obj_grad[ielem] += vKu * self.gradient_factor
        return obj_grad


    """
    [43]  compute_sensor_load

         Computes the difference between current and target displacements in each direction
         and returns it as sensor load for each sensor
    """
    def compute_sensor_load(self, icase, isample):
        sensor_load = np.zeros((self.ntime, self.nsensor, 3), dtype = np.float64)
        for itime in range(self.ntime):
            for isensor in range(self.nsensor):
                u = self.sensor_unkno_curr[icase][itime][isample][isensor][0]
                v = self.sensor_unkno_curr[icase][itime][isample][isensor][1]
                w = self.sensor_unkno_curr[icase][itime][isample][isensor][2]
                um = self.sensor_unkno_targ[icase][itime][isensor][0]
                vm = self.sensor_unkno_targ[icase][itime][isensor][1]
                wm = self.sensor_unkno_targ[icase][itime][isensor][2]
                du = u - um
                dv = v - vm
                dw = w - wm
                sensor_load[itime][isensor][0] += du * self.gradient_factor
                sensor_load[itime][isensor][1] += dv * self.gradient_factor
                sensor_load[itime][isensor][2] += dw * self.gradient_factor
        if self.obj_type == 0:
            sensor_load *= self.integ.weight[isample] * self.pdf(icase, isample, self.integ.gpoint[isample])
        elif self.obj_type == 1:
            sensor_load *= self.integ.weight[isample] * self.pdf(icase, isample, self.integ.gpoint[isample]) \
                * self.plus_func_dx(self.get_objective(icase, isample) - self.var) / (1.0 - self.cvar_beta)
        elif self.obj_type == 2:
            objval = self.get_objective(icase, isample)
            sensor_load *= self.integ.weight[isample] * self.pdf(icase, isample, self.integ.gpoint[isample]) \
                * (self.risk_lambda + (1.0-self.risk_lambda) * self.plus_func_dx(objval-self.var) / (1.0-self.cvar_beta))

        return sensor_load


    """
    [44]  apply_smoothing

         Performs smoothing over an element variable by moving the values to points
         averaged by how many elements belong to this point, and then moving the value
         back to elements. Performs nsmoothing passes of this process.
    """
    def apply_smoothing(self, nsmoothing, elem_var):
        npoin = self.npoin
        nelem = self.nelem
        for ipass in range(nsmoothing):
            var_at_points = np.zeros(npoin, dtype=np.float64)
            var_count = np.zeros(npoin, dtype=np.int32)
            for ielem in range(nelem):
                element = self.element[ielem]
                for ipoin in range(element.npoin):
                    ip = element.point[ipoin]
                    var_at_points[ip] += elem_var[ielem]
                    var_count[ip] += 1
            for ipoin in range(npoin):
                var_at_points[ipoin] = var_at_points[ipoin]/var_count[ipoin]
            for ielem in range(nelem):
                element = self.element[ielem]
                elem_var[ielem] = 0.0
                for ipoin in range(element.npoin):
                    ip = element.point[ipoin]
                    elem_var[ielem] += var_at_points[ip]
                elem_var[ielem] /= element.npoin
        return elem_var


    """
    [45]  compute_objective

         Modifies the elastic properties of the reference input file
         and runs Calculix to obtain the objective for current strength_factor
    """
    def compute_objective(self, icase, isample):
        # Read the reference file
        if self.nsample <= 1:
            ref_file = "ccx_input/calculix_forward_case" + str(icase)
        else:
            ref_file = "ccx_input/calculix_forward_case" + str(icase) + "_sample" + str(isample)
        with open(ref_file + '.inp', 'r') as file:
            lines = file.readlines()

        # Pattern to detect *MATERIAL sections and *ELASTIC lines
        material_pattern = re.compile(r'\*MATERIAL,NAME=MAT(\d+)')
        elastic_pattern = re.compile(r'\*ELASTIC')

        # Counter for materials
        material_count = 0

        # Process file lines
        for i, line in enumerate(lines):
            if elastic_pattern.search(line):  # If we find the *ELASTIC card
                # Fetch young modulus and poisson number from the next line
                sline = lines[i+1].split(',')
                young = float(sline[0])
                poisson = float(sline[1])
                # Replace the elastic property on the next line with the new one
                newline = f"{young*self.strength_factor[material_count]:16.8E}, {poisson:16.8E} \n"
                lines[i + 1] = newline
                material_count += 1

        # Writing the modified file to a new file
        new_file = 'results/calculix_forward_case' + str(icase) + '_sample' + str(isample)
        with open(new_file + '.inp', 'w') as file:
            file.writelines(lines)

        # Run CCX
        os.system('ccx ' + new_file + ' > ' + new_file + '.out')

        # Read unknown at sensors
        sensor_unkno = self.get_unknown_at_sensors(new_file + '.dat')

        # Return icase and unknown at sensors
        return icase, isample, sensor_unkno


    """
    [46]  compute_objective_target

         Modifies the elastic properties of the reference input file
         and runs Calculix to obtain the objective for a given target strength_factor
    """
    def compute_objective_target(self, icase, strength_factor):
        # Make target folder
        os.system('mkdir -p results/target')
        # Read the reference file
        ref_file = "ccx_input/calculix_forward_case" + str(icase)
        with open(ref_file + '.inp', 'r') as file:
            lines = file.readlines()

        # Pattern to detect *MATERIAL sections and *ELASTIC lines
        material_pattern = re.compile(r'\*MATERIAL,NAME=MAT(\d+)')
        elastic_pattern = re.compile(r'\*ELASTIC')

        # Counter for materials
        material_count = 0

        # Process file lines
        for i, line in enumerate(lines):
            if elastic_pattern.search(line):  # If we find the *ELASTIC card
                # Fetch young modulus and poisson number from the next line
                sline = lines[i+1].split(',')
                young = float(sline[0])
                poisson = float(sline[1])
                # Replace the elastic property on the next line with the new one
                newline = f"{young*strength_factor[material_count]:16.8E}, {poisson:16.8E} \n"
                lines[i + 1] = newline
                material_count += 1

        # Writing the modified file to a new file
        new_file = 'results/target/calculix_forward_case' + str(icase)
        with open(new_file + '.inp', 'w') as file:
            file.writelines(lines)

        # Run CCX
        os.system('ccx ' + new_file + ' > ' + new_file + '.out')
        # Generate VTK
        generate_vtk(new_file + '.frd')
        # Add strength factor to the vtk
        self.add_control_to_vtk(new_file, strength_factor)
        # Make sensor file
        self.generate_sensor_diff_vtk(icase, 0, new_file)

        # Read unknown at sensors
        sensor_unkno = self.get_unknown_at_sensors(new_file + '.dat')

        # Return icase, unknown at sensors
        return icase, sensor_unkno


    """
    [47]  compute_gradient

         Modifies the elastic properties of the reference input file
         and runs Calculix to obtain the gradient for a given target strength_factor
    """
    def compute_gradient(self, icase, isample):
        # Read the reference file
        if self.nsample <= 1:
            ref_file = "results/ref/calculix_adjoint_case" + str(icase)
        else:
            ref_file = "results/ref/calculix_adjoint_case" + str(icase) + "_sample" + str(isample)
        with open(ref_file + '.inp', 'r') as file:
            lines = file.readlines()

        # Pattern to detect *MATERIAL sections and *ELASTIC lines
        material_pattern = re.compile(r'\*MATERIAL,NAME=MAT(\d+)')
        elastic_pattern = re.compile(r'\*ELASTIC')
        cload_pattern = re.compile(r'\*CLOAD')

        # Counter for materials
        material_count = 0

        # Retrieve sensor load
        sensor_load = self.compute_sensor_load(icase, isample)

        # Process file lines
        icload = -1
        for i, line in enumerate(lines):
            if elastic_pattern.search(line):  # If we find the *ELASTIC card
                # Fetch young modulus and poisson number from the next line
                sline = lines[i+1].split(',')
                young = float(sline[0])
                poisson = float(sline[1])
                # Replace the elastic property on the next line with the new one
                newline = f"{young*self.strength_factor[material_count]:16.8E}, {poisson:16.8E} \n"
                lines[i + 1] = newline
                material_count += 1
            if cload_pattern.search(line):
                if self.run_type == 'dynamic':
                    sline   = line.split('=')
                    itime   = int(''.join([char for char in sline[1] if char.isdigit()]))
                    sline   = lines[i+1].split(',')
                    isensor = int(''.join([char for char in sline[0] if char.isdigit()]))
                    newline = f"LOAD{isensor}, 1, {sensor_load[itime][isensor][0]:16.8E}\n"
                    lines[i + 1] = newline
                    newline = f"LOAD{isensor}, 2, {sensor_load[itime][isensor][1]:16.8E}\n"
                    lines[i + 2] = newline
                    newline = f"LOAD{isensor}, 3, {sensor_load[itime][isensor][2]:16.8E}\n"
                    lines[i + 3] = newline
                else:
                    icload += 1
                    sline = lines[i+1].split(',')
                    ipoin = int(''.join([char for char in sline[0] if char.isdigit()]))
                    newline = f"LOAD{ipoin}, 1, {sensor_load[0][icload][0]:16.8E}\n"
                    lines[i + 1] = newline
                    newline = f"LOAD{ipoin}, 2, {sensor_load[0][icload][1]:16.8E}\n"
                    lines[i + 2] = newline
                    newline = f"LOAD{ipoin}, 3, {sensor_load[0][icload][2]:16.8E}\n"
                    lines[i + 3] = newline

        # Writing the modified file to a new file
        new_file = 'results/calculix_adjoint_case' + str(icase) + '_sample' + str(isample)
        with open(new_file + '.inp', 'w') as file:
            file.writelines(lines)

        # Run CCX
        os.system('ccx ' + new_file + ' > ' + new_file + '.out')

        # Return gradient
        return self.get_gradient(icase, isample)


    """
    [48]  plus_func

         Plus function: f(x) = x if x > 0
                               0 else
    """
    def plus_func(self, x):
        eps = 1.0e-16
        if x > eps:
            g = x
        elif x < -eps:
            g = 0.0
        else:
            g = eps
        return g


    """
    [49]  plus_func_dt

         Derivative of the plus function
    """
    def plus_func_dx(self, x):
        eps = 1.0e-16
        if x > eps:
            gdx = 1.0
        elif x < -eps:
            gdx = 0.0
        else:
            gdx = eps
        return gdx


    """
    [50]  pdf

         Derivative of the plus function
    """
    def pdf(self, icase, isample, x):
        pdf = 1.0
        return pdf


    """
    [51]  compute_var_finite_diff

         Uses finite difference to compute the gradient of the objective function
         with respect to the VaR. For CVaR runs.
    """
    def compute_var_finite_diff(self):
        delta = 1.0e-04

        var2 = self.var+delta
        if self.var-delta < self.var_min:
            var1 = self.var_min
            delta = var2-var1
        else:
            var1 = self.var-delta

        obj1 = 0.0
        obj2 = 0.0

        for icase in range(self.ncase):
            for isample in range(self.nsample):
                # Store new objective
                if self.obj_type == 1:
                    objval = self.get_objective(icase, isample)
                    obj1 += self.integ.weight[isample] * self.pdf(icase, isample, self.integ.gpoint[isample]) \
                         * (self.plus_func(objval-var1) / (1.0-self.cvar_beta) + var1)
                    obj2 += self.integ.weight[isample] * self.pdf(icase, isample, self.integ.gpoint[isample]) \
                         * (self.plus_func(objval-var2) / (1.0-self.cvar_beta) + var2)
                elif self.obj_type == 2:
                    objval = self.get_objective(icase, isample)
                    obj1 += self.integ.weight[isample] * self.pdf(icase, isample, self.integ.gpoint[isample]) \
                        * (self.risk_lambda*objval \
                        + (1.0-self.risk_lambda) * (self.plus_func(objval-var1) / (1.0-self.cvar_beta) + var1))
                    obj2 += self.integ.weight[isample] * self.pdf(icase, isample, self.integ.gpoint[isample]) \
                        * (self.risk_lambda*objval \
                        + (1.0-self.risk_lambda) * (self.plus_func(objval-var2) / (1.0-self.cvar_beta) + var2))

        with open('var.dat', "a") as f:
            f.write(f"{self.var:16.8E}, {(obj2-obj1)/(2*delta):16.8E}\n")

        return (obj2-obj1)/(2*delta)


    """
    [52]  compute_sample_adjoint

        Computes a Calculix adjoint run to have a reference input file 
        results/ref/calculix_adjoint_case(#). This file will be copied in all forward runs
        and its materials properties and loads modified.
    """
    def compute_sample_adjoint(self):
        # Make directories
        os.system('mkdir -p results')
        os.system('mkdir -p results/ref')
        # Make input files
        for icase in range(self.ncase):
            filename = 'results/ref/calculix_adjoint_case' + str(icase)
            with open(filename + '.inp', 'w') as fu:
                fu.write('**                          \n')
                fu.write('** Calculix adjoint problem \n')
                fu.write('**                          \n')
                fu.write('*NODE,NSET=NALL\n')
                for ipoin in range(self.npoin):
                    fu.write(f"{ipoin+1}, ")
                    fu.write(f"{self.point_coord[ipoin][0]:16.8E}, ")
                    fu.write(f"{self.point_coord[ipoin][1]:16.8E}, ")
                    fu.write(f"{self.point_coord[ipoin][2]:16.8E}\n")
                for ielem in range(self.nelem):
                    fu.write(f'*ELEMENT,TYPE={self.element[ielem].element_type},ELSET=E{ielem}\n')
                    fu.write(f"{ielem+1},")
                    for ipoin in range(self.element[ielem].npoin):
                        fu.write(f"{self.element[ielem].point[ipoin]+1},")
                    fu.write('\n')
                fu.write('*ELSET,ELSET=EALL\n')
                for ielem in range(self.nelem):
                    fu.write(f"{ielem+1}\n")
                fu.write('*BOUNDARY\n')
                for ipoin in self.boundary_type:
                    if self.boundary_type[ipoin] == 0:
                        fu.write(f"{ipoin}, {self.boundary_dofs[ipoin]}\n")
                    elif self.boundary_type[ipoin] == 1:
                        fu.write(f"{ipoin}, 1, 1\n")
                    elif self.boundary_type[ipoin] == 2:
                        fu.write(f"{ipoin}, 2, 2\n")
                    elif self.boundary_type[ipoin] == 3:
                        fu.write(f"{ipoin}, 3, 3\n")
                    elif self.boundary_type[ipoin] == 4:
                        fu.write(f"{ipoin}, 1, 3\n")
                    elif self.boundary_type[ipoin] == 5:
                        fu.write(f"{ipoin}, 4, 6\n")
                    elif self.boundary_type[ipoin] == 6:
                        fu.write(f"{ipoin}, 1, 6\n")
                for ielem in range(self.nelem):
                    fu.write(f'*MATERIAL,NAME=MAT{ielem}\n')
                    fu.write('*ELASTIC\n')
                    fu.write(f"{self.element[ielem].young_modulus:16.8E}, ")
                    fu.write(f"{self.element[ielem].poisson_coef}\n")
                    fu.write('*DENSITY\n')
                    fu.write(f"{self.element[ielem].density:16.8E}\n")
                for ielem in range(self.nelem):
                    if self.element[ielem].section_type.strip() == 'SOLID':
                        fu.write(f"*SOLID SECTION,ELSET=E{ielem},MATERIAL=MAT{ielem}\n")
                    elif self.element[ielem].section_type.strip() == 'RECT':
                        fu.write(f"*BEAM SECTION,ELSET=E{ielem},MATERIAL=MAT{ielem},SECTION=RECT\n")
                        fu.write(f"{self.element[ielem].beam_width:16.8E}, {self.element[ielem].beam_height:16.8E}\n")
                        first_normal = self.get_beam_first_normal(ielem)
                        fu.write(f"{first_normal[0]:16.8E}, {first_normal[1]:16.8E}, {first_normal[2]:16.8E}\n")
                    elif self.element[ielem].section_type.strip() == 'BOX':
                        fu.write(f"*BEAM SECTION,ELSET=E{ielem},MATERIAL=MAT{ielem},SECTION=BOX\n")
                        fu.write(f"{self.element[ielem].beam_width:16.8E}, {self.element[ielem].beam_height:16.8E},")
                        fu.write(f"{self.element[ielem].beam_t1:16.8E}, {self.element[ielem].beam_t2:16.8E},")
                        fu.write(f"{self.element[ielem].beam_t3:16.8E}, {self.element[ielem].beam_t4:16.8E}\n")
                        first_normal = self.get_beam_first_normal(ielem)
                        fu.write(f"{first_normal[0]:16.8E}, {first_normal[1]:16.8E}, {first_normal[2]:16.8E}\n")
                    elif self.element[ielem].section_type.strip() == 'SHELL':
                        fu.write(f"*SHELL SECTION,ELSET=E{ielem},MATERIAL=MAT{ielem}\n")
                        fu.write(f"{self.element[ielem].shell_thickness:16.8E}\n")
                    else:
                        print('Element section type', self.element[ielem].section_type.strip(), 'unrecognized')
                        raise ValueError('Element section type unrecognized')
                for isensor in range(self.nsensor):
                    fu.write(f"*NSET,NSET=LOAD{isensor}\n")
                    fu.write(f"{self.sensor_point[isensor]+1}\n")
                if self.run_type == 'dynamic':
                    fu.write('*TIMEPOINTS,NAME=TPRINT,GENERATE\n')
                    fu.write(f"0.0E+00,{self.dt*(self.ntime-1):16.8E},{self.dt:16.8E}\n")
                    fu.write('*STEP\n')
                    fu.write(f"*DYNAMIC,ALPHA=-0.3,SOLVER=ITERATIVE CHOLESKY,DIRECT\n")
                    fu.write(f"1.0E-10,1.0E-10,1.0E-10,1.0E-10\n")
                    fu.write(f"*AMPLITUDE,NAME=A{self.ntime-1}AUX\n")
                    fu.write(f"0.0E+00, 1.0E+00, 1.0E-10, 0.0E+00\n")
                    for isensor in range(self.nsensor):
                        fu.write(f"*CLOAD,AMPLITUDE=A{self.ntime-1}AUX\n")
                        fu.write(f"LOAD{isensor}, 1, {0.0:16.8E}\n")
                        fu.write(f"LOAD{isensor}, 2, {0.0:16.8E}\n")
                        fu.write(f"LOAD{isensor}, 3, {0.0:16.8E}\n")
                    fu.write('*NODE PRINT,NSET=NAll\n')
                    fu.write('U\n')
                    fu.write('*NODE FILE,OUTPUT=3D\n')
                    fu.write('U\n')
                    fu.write('*END STEP\n')
                    for itime in range(self.ntime-1):
                        fu.write('*STEP,INC=10000\n')
                        if itime == 0:
                            fu.write(f"*DYNAMIC,ALPHA=-0.3,SOLVER=ITERATIVE CHOLESKY\n")
                            fu.write(f"{self.dt-1.0E-10},{self.dt-1.0E-10}\n")
                            fu.write(f"*AMPLITUDE,NAME=A{self.ntime-1-itime}\n")
                            #fu.write(f"0.0E+00, 0.5E+00, {0.5*self.dt:16.8E}, 1.0E+00, {self.dt-1.0E-10:16.8E}, 0.5E+00\n")
                            fu.write(f"0.0E+00, 1.0E+00, {self.dt:16.8E}, 1.0E+00\n")
                        else:
                            fu.write(f"*DYNAMIC,ALPHA=-0.3,SOLVER=ITERATIVE CHOLESKY\n")
                            fu.write(f"{self.dt},{self.dt}\n")
                            fu.write(f"*AMPLITUDE,NAME=A{self.ntime-1-itime}\n")
                            #fu.write(f"0.0E+00, 1.0E+00, {0.5*self.dt:16.8E}, 1.0E+00, {self.dt:16.8E}, 0.5E+00\n")
                            fu.write(f"0.0E+00, 1.0E+00, {self.dt:16.8E}, 1.0E+00\n")
                        for isensor in range(self.nsensor):
                            fu.write(f"*CLOAD,AMPLITUDE=A{self.ntime-1-itime}\n")
                            fu.write(f"LOAD{isensor}, 1, {0.0:16.8E}\n")
                            fu.write(f"LOAD{isensor}, 2, {0.0:16.8E}\n")
                            fu.write(f"LOAD{isensor}, 3, {0.0:16.8E}\n")
                        fu.write('*NODE PRINT,NSET=NAll,TIME POINTS=TPRINT\n')
                        fu.write('U\n')
                        fu.write('*NODE FILE,OUTPUT=3D,TIME POINTS=TPRINT\n')
                        fu.write('U\n')
                        fu.write('*END STEP\n')
                elif self.run_type == 'static':
                    fu.write('*STEP\n')
                    fu.write('*STATIC\n')
                    for isensor in range(self.nsensor):
                        fu.write('*CLOAD\n')
                        fu.write(f"LOAD{isensor}, 1, {0.0:16.8E}\n")
                        fu.write(f"LOAD{isensor}, 2, {0.0:16.8E}\n")
                        fu.write(f"LOAD{isensor}, 3, {0.0:16.8E}\n")
                    fu.write('*NODE PRINT,NSET=NAll\n')
                    fu.write('U\n')
                    fu.write('*NODE FILE,OUTPUT=3D\n')
                    fu.write('U\n')
                    fu.write('*END STEP\n')


    """
    [53]  add_results_to_movie

         Generates vtk results and adds them to results/movie
    """
    def add_results_to_movie(self):
        self.iter += 1
        if (self.iter == 0): os.system('mkdir -p results/movie')
        with ProcessPoolExecutor(max_workers=self.nproc_cases) as executor:
            run_futures = [executor.submit(self.add_results_to_movie_serial, icase) \
                           for icase in range(self.ncase)]
            for future in as_completed(run_futures):
                try:
                    result = future.result()
                except Exception as exc:
                    print(f'Exception in add_results_to_movie_random_serial: {exc}')


    """
    [54]  add_results_to_movie_serial

         Generates vtk results and adds them to results/movie
    """
    def add_results_to_movie_serial(self, icase):
        # Generate VTKs
        isample = 0
        filename_fwd = 'results/calculix_forward_case' + str(icase) + '_sample' + str(isample)
        filename_adj = 'results/calculix_adjoint_case' + str(icase) + '_sample' + str(isample)
        # Copy into movie
        if self.ntime <= 1:
            generate_vtk(filename_fwd + '.frd')
            self.add_control_to_vtk(filename_fwd, self.strength_factor)
            self.generate_sensor_diff_vtk(icase, isample, filename_fwd)
            os.system('cp ' + filename_fwd + '.vtk results/movie/forward_case' + str(icase) + '_iter' + str(self.iter) + '.vtk')
            os.system('cp ' + filename_fwd + '_sensor_diff.vtk results/movie/forward_case' + str(icase) \
                      + '_sensor_diff_iter' + str(self.iter) + '.vtk')
            os.system('cp ' + filename_adj + '.vtk results/movie/adjoint_case' + str(icase) + '_iter' + str(self.iter) + '.vtk')
        else:
            generate_vtk(filename_fwd + '.frd')
            self.add_control_to_vtk(filename_fwd, self.strength_factor)
            if self.iter%10 == 0:
                os.system('mkdir -p results/dynamic_movie_iter' + str(self.iter))
                self.generate_sensor_diff_vtk(icase, isample, filename_fwd)
                os.system('cp results/*.vtk results/dynamic_movie_iter' + str(self.iter) + '/.')
            ndigit = len(str(self.ntime))
            os.system('cp ' + filename_fwd + f".{self.ntime:0{ndigit}}" + '.vtk results/movie/forward_case' \
                      + str(icase) + '_iter' + str(self.iter) + '.vtk')
            os.system('cp ' + filename_adj + f".{self.ntime:0{ndigit}}" + '.vtk results/movie/adjoint_case' \
                      + str(icase) + '_iter' + str(self.iter) + '.vtk')


    """
    [55]  target

         Computes the target cases and stores the target displacements
    """
    def target(self, strength_factor):
        # Run target cases
        with ProcessPoolExecutor(max_workers=self.nproc_cases) as executor:
            run_futures = [executor.submit(self.compute_objective_target, icase, strength_factor) \
                           for icase in range(self.ncase)]
            for future in as_completed(run_futures):
                try:
                    icase, sensor_unkno = future.result()
                    self.sensor_unkno_targ[icase] = sensor_unkno
                except Exception as exc:
                    print(f'Exception in compute_objective_target: {exc}')

        self.compute_sample_adjoint()


    """
    [56]  objective

         Computes and returns the objective function for every case
    """
    def objective(self):
        obj = 0.0
        with ProcessPoolExecutor(max_workers=self.nproc_cases) as executor:
            run_futures = {executor.submit(self.compute_objective, icase, isample): \
                           (icase, isample) for icase, isample in product(range(self.ncase), range(self.nsample))}
            for future in as_completed(run_futures):
                try:
                    icase, isample, sensor_unkno = future.result()
                    for itime in range(self.ntime):
                        self.sensor_unkno_curr[icase][itime][isample] = sensor_unkno[itime]
                    if self.obj_type == -1:
                        obj += self.get_objective(icase, isample)
                    elif self.obj_type == 0:
                        obj += self.integ.weight[isample] * self.pdf(icase, isample, self.integ.gpoint[isample]) \
                            * self.get_objective(icase, isample)
                    elif self.obj_type == 1:
                        obj += self.integ.weight[isample] * self.pdf(icase, isample, self.integ.gpoint[isample]) \
                            * ( self.plus_func(self.get_objective(icase, isample)-self.var) / (1.0-self.cvar_beta) \
                                + self.var)
                    elif self.obj_type == 2:
                        objval = self.get_objective(icase, isample)
                        obj += self.integ.weight[isample] * self.pdf(icase, isample, self.integ.gpoint[isample]) \
                            * (self.risk_lambda*objval + (1.0-self.risk_lambda) * (self.plus_func(objval-self.var) \
                              / (1.0-self.cvar_beta) + self.var))
                except Exception as exc:
                    print(f'Exception in compute_objective: {exc}')

        if self.obj_type in [1, 2]:
            self.var -= 1.0e-03*self.compute_var_finite_diff()
            if self.var < self.var_min: self.var = self.var_min

        return obj


    """
    [57]  gradient

         Computes and returns the gradient function for every case
    """
    def gradient(self):
        obj_grad = np.zeros(self.nelem, dtype=np.float64)
        with ProcessPoolExecutor(max_workers=self.nproc_cases) as executor:
            run_futures = {executor.submit(self.compute_gradient, icase, isample): \
                           (icase, isample) for icase, isample in product(range(self.ncase), range(self.nsample))}
            for future in as_completed(run_futures):
                try:
                    obj_grad += future.result()
                except Exception as exc:
                    print(f'Exception in compute_gradient: {exc}')

        for icase in range(self.ncase):
            isample = 0
            filename_adj = 'results/calculix_adjoint_case' + str(icase) + '_sample' + str(isample)
            generate_vtk(filename_adj + '.frd')
            self.add_grad_to_vtk(filename_adj, obj_grad, smooth=False)

        if self.nsmoothing > 0:
            self.apply_smoothing(self.nsmoothing, obj_grad)

        self.add_grad_to_vtk(filename_adj, obj_grad)

        return obj_grad


    """
    [58]  print_control_space

         Receives an array of element numbers. Outputs 1 if element is present, 0 otherwise.
    """
    def print_control_space(self, control):
        os.system('cp results/calculix_forward_case0_sample0.vtk results/control_space.vtk')
        with open('results/control_space.vtk', 'a') as fu:
            fu.write('\n')
            fu.write('SCALARS control_space double 1\n')
            fu.write('LOOKUP_TABLE my_table\n')
            for icontrol in range(self.nelem):
                if icontrol in control:
                    fu.write(f"{1.0}\n")
                else:
                    fu.write(f"{0.0}\n")


    """
    [59]  print_strfac_comparison

         Given two values of strength factor, prints the absolute difference in a vtk file.
    """
    def print_strfac_comparison(self, strfac0, strfac1, filename):
        runstr = 'cp results/calculix_forward_case0_sample0.vtk results/' + filename + '.vtk'
        os.system(runstr)
        with open('results/' + filename +'.vtk', 'a') as fu:
            fu.write('\n')
            fu.write('SCALARS abs_diff double 1\n')
            fu.write('LOOKUP_TABLE my_table\n')
            for icontrol in range(self.nelem):
                fu.write(f"{np.abs(strfac0[icontrol]-strfac1[icontrol])}\n")
