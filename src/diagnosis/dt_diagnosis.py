import numpy as np
from digital_twin import *
from src.model.dt_model import *
from pyrol import getCout, Objective, Problem, Solver, getParametersFromXmlFile, Bounds
from pyrol.vectors import NumPyVector as myVector
import copy
import scipy.optimize as opt
from scipy.linalg import cho_solve, cho_factor
from scipy.optimize import minimize
import sys
import os

def system_identification(model, input_vars={}, pyrol_file=None, output_file=None):
    if pyrol_file is not None:

        class opt_objective(Objective):
            def __init__(self):
                super().__init__()

            def value(self, control, tol):
                model.update_strength_factor(control)
                obj = copy.copy(model.objective())
                return obj

            def gradient(self, grad, control, tol):
                model.update_strength_factor(control)
                g = model.gradient()
                for ielem in range(model.nelem):
                    grad[ielem] = copy.copy(g[ielem])
                model.add_results_to_movie()

        # Read inputs
        lower_limit = input_vars.get('lower_limit', 1.0e-01)
        upper_limit = input_vars.get('upper_limit', 1.0e+00)

        # Configure parameter list.
        params = getParametersFromXmlFile(pyrol_file)

        # Set the output stream.
        stream = getCout()

        if output_file is not None:
            # Redirect stdout and stderr
            log_file = open(output_file, "w")
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(log_file.fileno(), sys.stdout.fileno())  # Redirect stdout
            os.dup2(log_file.fileno(), sys.stderr.fileno())  # Redirect stderr

        # Set bounds.
        lower_bounds = myVector(np.full(model.nelem, lower_limit))
        upper_bounds = myVector(np.full(model.nelem, upper_limit))
        bounds = Bounds(lower_bounds, upper_bounds)

        # Control and gradient variables
        control = myVector(np.ones(model.nelem, dtype=np.float64))
        for icontrol in range(model.nelem):
            control[icontrol] = model.strength_factor[icontrol]
        grad = control.dual()

        # Set up the problem.
        objective = opt_objective()
        problem = Problem(objective, control, grad)
        problem.addBoundConstraint(bounds)

        # Solve.
        solver = Solver(problem, params)
        solver.solve(stream)

        if output_file is not None:
            # Restore stdout/stderr
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__

            # Close the log file
            log_file.close()

    else:

        def value(control):
            model.update_strength_factor(control)
            obj = copy.copy(model.objective())
            return obj

        def gradient(control):
            model.update_strength_factor(control)
            grad = model.gradient()
            model.add_results_to_movie()
            return grad

        # Read inputs
        lower_limit = input_vars.get('lower_limit', 1.0e-01)
        upper_limit = input_vars.get('upper_limit', 1.0)

        # Bounds
        lower_bounds = np.full(model.nelem, lower_limit)
        upper_bounds = np.full(model.nelem, upper_limit)
        bounds = opt.Bounds(lower_bounds, upper_bounds)

        # Control and gradient variables
        control = np.ones(model.nelem, dtype=np.float64)
        for icontrol in range(model.nelem):
            control[icontrol] = model.strength_factor[icontrol]
        grad = np.zeros(model.nelem, dtype=np.float64)

        if output_file is not None:
            # Duplicate stdout and stderr
            log_file = open(output_file, "w")
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(log_file.fileno(), sys.stdout.fileno())  # Redirect stdout
            os.dup2(log_file.fileno(), sys.stderr.fileno())  # Redirect stderr

        res = minimize(value, control, method='L-BFGS-B', jac=gradient, bounds=bounds, \
                       options={'gtol': 1e-12, 'disp': True})

        if output_file is not None:
            # Restore stdout/stderr
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__

            # Close the log file
            log_file.close()
